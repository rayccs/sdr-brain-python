"""
SDR Brain v2.0 — Motor Cognitivo del Agente de Ventas
──────────────────────────────────────────────────────
Responsabilidades:
1. Recibir el mensaje del prospecto + historial de conversación
2. Generar una respuesta empática y estratégica (SDR experto en B2B)
3. Clasificar el lead con score BANT (0-100)
4. Detectar el dolor del lead y el estado de calificación
5. Retornar respuesta + clasificación al orquestador N8N
"""

import os
import io
import json
import base64
import logging
import re
from collections import Counter
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import PyPDF2
from docx import Document

import niche_presets

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# App & CORS
# ──────────────────────────────────────────────────────────────────
app = FastAPI(title="SDR Brain API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, ResponseValidationError

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global Exception: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Unhandled Exception: {str(exc)}"}
    )

@app.exception_handler(ResponseValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Response Validation Error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": f"Response Validation Error: {str(exc)}"}
    )


# ──────────────────────────────────────────────────────────────────
# LLM — usa OpenRouter con modelo configurable
# ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Usamos Hermes 3 por defecto para máxima obediencia y zero censura (Ideal para SDR B2B)
MODEL = os.getenv("LLM_MODEL", "nousresearch/hermes-3-llama-3.1-405b")

def get_llm(model_override: Optional[str] = None, temperature: float = 0.6, max_tokens: int = 1024) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_override or MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )

def get_gemini_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="google/gemini-2.5-flash",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=1024,
    )

# ──────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────
class ConversationMessage(BaseModel):
    role: str       # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    """
    Payload que envía N8N al Brain con toda la información necesaria.
    """
    message: str                                 # Último mensaje del prospecto
    history: Optional[List[ConversationMessage]] = []  # Historial completo
    company_config: Optional[dict] = None        # Config del cliente (ICP, oferta, prompt)
    lead_phone: Optional[str] = None
    lead_name: Optional[str] = None
    lead_status: Optional[str] = None
    company_id: Optional[str] = "default_company"

class ClassifyRequest(BaseModel):
    """Para clasificar un lead solo con el historial sin generar respuesta"""
    history: List[ConversationMessage]
    company_config: Optional[dict] = None

class BantScore(BaseModel):
    budget: str
    authority: str
    need: str
    timeline: str
    score: int                  # 0-100
    status: str                 # EN_CALIFICACION | POR_AGENDAR | DESCALIFICADO | EN_SEGUIMIENTO
    pain: str
    summary: str
    name: str
    interest: Optional[str] = None
    objections: Optional[str] = None
    next_step: Optional[str] = None
    strategy: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    bant: BantScore
    apollo_insights: Optional[str] = None
    model_used: str
    tokens_used: int = 0

class NexusChatRequest(BaseModel):
    query: str
    company_config: Optional[dict] = None
    leads_summary: Optional[List[dict]] = []
    conversations_sample: Optional[List[dict]] = []

class NexusChatResponse(BaseModel):
    action: str
    detail: str
    suggestion: str
    color: str
    category: Optional[str] = "GENERAL"
    findings: Optional[List[str]] = []
    recommended_action: Optional[str] = None
    target_tab: Optional[int] = 7

class ValidateConfigRequest(BaseModel):
    niche: str
    offer: str
    services: str

class ValidateConfigResponse(BaseModel):
    valid: bool
    reason: str

class HandoffAnalysisRequest(BaseModel):
    history: List[ConversationMessage]
    lead_phone: Optional[str] = None
    lead_name: Optional[str] = None
    company_config: Optional[dict] = None

class HandoffAnalysisResponse(BaseModel):
    bant_interest: str
    bant_budget: str
    bant_authority: str
    bant_need: str
    bant_timeline: str
    sentiment: str
    brief: str
    recommended_action: str
    model_used: str

# ──────────────────────────────────────────────────────────────────
# Lightweight In-Memory RAG
# ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """Divide el texto en párrafos para el RAG."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
        else:
            current_chunk += p + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def retrieve_relevant_chunks(query: str, text: str, top_k: int = 3) -> str:
    """Extrae los N fragmentos más relevantes del knowledge base según el query."""
    if not text or len(text) < 4000:
        return text # Si el texto es pequeño, devolver completo
        
    chunks = chunk_text(text)
    # Extracción simple de palabras (minúsculas, alfanumérico)
    query_words = set(re.findall(r'\w+', query.lower()))
    if not query_words:
        # Si el query no tiene palabras, devolver los primeros fragmentos
        return "\n\n...\n\n".join(chunks[:top_k])
    
    scored_chunks = []
    for chunk in chunks:
        chunk_words = re.findall(r'\w+', chunk.lower())
        word_counts = Counter(chunk_words)
        # Score = suma de la frecuencia de las palabras del query en el chunk
        score = sum(word_counts[w] for w in query_words)
        scored_chunks.append((score, chunk))
        
    # Ordenar por score descendente
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Devolver los top_k chunks unidos
    top_chunks = [c for score, c in scored_chunks[:top_k]]
    return "\n\n[...]\n\n".join(top_chunks)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def build_system_prompt(company_config: Optional[dict], user_query: str = "", lead_name: str = "") -> str:
    """
    Construye el system prompt del agente SDR basado en la configuración
    de la empresa cliente. Si no hay config, usa valores genéricos.
    """
    if company_config:
        company_name = company_config.get("name", company_config.get("Name", "nuestra empresa"))
        icp = company_config.get("icp", company_config.get("ICP", "empresas B2B"))
        offer = company_config.get("value_offer", company_config.get("ValueOffer", "soluciones de automatización"))
        services = company_config.get("services", company_config.get("Services", ""))
        custom_prompt = company_config.get("prompt", company_config.get("Prompt", ""))
        agent_name = company_config.get("agent_name", company_config.get("AgentName", "Alex"))
        knowledge_base = company_config.get("knowledge_base", company_config.get("KnowledgeBase", ""))
    else:
        company_name = "nuestra empresa"
        icp = "empresas B2B medianas y grandes"
        offer = "soluciones de automatización e inteligencia artificial"
        services = "Agentes IA, automatizaciones de ventas, CRM inteligente"
        custom_prompt = ""
        agent_name = "Alex"
        knowledge_base = ""
        
    kb_section = ""
    resources_section = ""
    if knowledge_base:
        parts = knowledge_base.split("## RECURSOS PARA COMPARTIR CON EL LEAD (Catálogos, Links):")
        base_kb = parts[0]
        if len(parts) > 1:
            resources_section = "\n\n## RECURSOS PARA COMPARTIR CON EL LEAD (Catálogos, Links):" + parts[1]
            
        relevant_kb = retrieve_relevant_chunks(user_query, base_kb)
        kb_section = f"\n\n## BASE DE CONOCIMIENTO EXTRAÍDA DE DOCUMENTOS:\nUsa esta información técnica y comercial para responder preguntas detalladas sobre los servicios/productos de {company_name}. Esta es información filtrada según lo que preguntó el cliente:\n{relevant_kb}\n\n"

    apollo_enabled = company_config.get("apollo_enabled", False)
    apollo_insights = company_config.get("apollo_insights", "")

    apollo_section = ""
    if apollo_enabled and not apollo_insights:
        apollo_section = "\n- **Misión de Enriquecimiento (Apollo):** Aún no tenemos el correo corporativo del prospecto. Pídeselo de manera MUY natural y sutil en tu respuesta (ej. para enviarle un catálogo, registrarlo en el sistema o validar su perfil corporativo). Solo hazlo si la conversación se presta para ello."
    elif apollo_insights:
        apollo_section = f"\n- **Datos Extraídos (Inteligencia Comercial):** {apollo_insights} -> Usa esta información sutilmente para hiper-personalizar tu respuesta y empatizar con el cargo del cliente."

    # --- Niche Presets ---
    preset = niche_presets.get_niche_preset(icp)
    preset_tone = preset.get("tone", "Profesional y claro.")
    preset_length = preset.get("length_rule", "MÁXIMO 2 a 5 líneas por mensaje.")
    preset_handoff = preset.get("handoff_rule", "Ofrece agendar reunión.")

    auto_schedule = company_config.get("auto_schedule_enabled", False)
    access_token = company_config.get("google_access_token", "")
    business_hours = company_config.get("business_hours", "Lunes a Viernes, 09:00 - 18:00")
    calendar_section = ""
    if auto_schedule and access_token:
        calendar_section = f"""

## AGENDAMIENTO AUTÓNOMO (Google Calendar) — OBLIGATORIO
TIENES herramientas (tools) de calendario. Tu comportamiento OBLIGATORIO es:

REGLA ABSOLUTA: Cuando el cliente dice "sí", "quiero agendar", "dale", "por favor", 
"confirmo" o cualquier variante afirmativa a una propuesta de reunión, 
DEBES EJECUTAR la tool `check_availability` con la fecha de MAÑANA (o la fecha 
que el cliente sugiera). NO respondas con texto sugiriendo agendar sin usar la tool.

Horario de Atención de la Empresa: {business_hours}
Solo puedes ofrecer horarios que estén DENTRO de este horario de atención.

FLUJO IMPERATIVO:
1. Cliente muestra interés → USA `check_availability` inmediatamente.
2. Con el resultado, ofrece 2 opciones libres dentro del horario de atención.
3. Cliente confirma horario → USA `book_appointment`.
4. Si no tienes email del cliente, pídelo ANTES de agendar.
5. PROHIBIDO: Responder "¿Te gustaría agendar?" si el cliente ya dijo que sí. EJECUTA LA TOOL.

### ⚠️ REGLAS DE SEGURIDAD Y ANTI-ALUCINACIÓN (CRÍTICAS):
- **NUNCA inventes horarios.** Solo usa los horarios que devuelva la tool `check_availability`.
- **NUNCA ofrezas ir a la oficina** si no está explícitamente definido en tu Base de Conocimiento.
- **NUNCA envíes correos electrónicos** por tu cuenta. Si necesitas que el cliente reciba una invitación, di que un ejecutivo se encargará de eso.
- **SI LA TOOL FALLA** (token expirado, error 401, error técnico, o no hay disponibilidad real):
  → **INMEDIATAMENTE** di: "Tuvimos un inconveniente técnico para agendar. Un ejecutivo de nuestra empresa se pondrá en contacto contigo para coordinar tu reunión personalmente. ¿Te parece bien?"
  → **NUNCA** digas "te enviaré un correo", "intentaré de nuevo", o "vamos a nuestra oficina". 
  → **NUNCA** repitas disculpas o intentes agendar de nuevo. Deriva al ejecutivo."""
    else:
        calendar_section = f"""

## AGENDAMIENTO COMERCIAL (PLAN ESTÁNDAR — SIN GOOGLE CALENDAR) — OBLIGATORIO
Google Calendar NO está disponible en este plan. NO tienes herramientas de agendamiento automático ni debes ofrecerlo.
- **PROHIBIDO:** Decir "te enviaré un correo", "te agendaré en Google Calendar", o "te generé un link de Google Meet".
- **PROHIBIDO:** Prometer enlaces o confirmaciones directas de calendario.
- **FLUJO CORRECTO:** Cuando el prospecto muestre interés en una reunión o llamada:
  1. Pregúntale qué día y hora le acomoda dentro del horario de atención de la empresa ({business_hours}).
  2. Confirma o solicita su nombre y correo corporativo/teléfono para el registro.
  3. Indícale que un **ejecutivo comercial de nuestra empresa** se pondrá en contacto de inmediato para coordinar los accesos y confirmar la reunión.
"""

    catalog_instruction = ""
    if resources_section and resources_section.strip():
        catalog_instruction = "\n5. **Recursos y Catálogos (PROACTIVIDAD):** En la sección de recursos tienes catálogos o links. Sé proactivo. Si el cliente pregunta por productos o precios, dile que tienes un catálogo y pregunta si quiere que se lo envíes. NUNCA INVENTES LINKS, usa EXCLUSIVAMENTE los links exactos proporcionados en los recursos."
    else:
        catalog_instruction = "\n5. **Carencia de Catálogos (HUMILDAD EPISTÉMICA):** NO tienes catálogos, brochures, links de casas, ni listas de precios pre-armadas en tus recursos. Si el cliente te pide un catálogo, brochure o link, RESPONDERÁS QUE NO TIENES ESA INFORMACIÓN y ofrecerás agendar una llamada o transferirlo con un ejecutivo. ESTRICTAMENTE PROHIBIDO inventar links o decir '[Link al catálogo]'."

    base_prompt = f"""Eres {agent_name}, un asesor experto, claro, directo y profesional que representa a "{company_name}". 
IMPORTANTE: Nunca menciones que eres una IA o un bot. Comórtate siempre como un humano natural, aplicando los principios de Humildad Epistémica (reconoce cordialmente si no sabes algo en lugar de inventarlo o alucinar) y Entropía Negativa (aporta siempre estructura, claridad y valor a la conversación del cliente en lugar de generar ruido). Usa RAG estrictamente basado en tu Base de Conocimiento.

## Conocimiento de Negocio del Cerebro de Ventas ({company_name})
- **Cliente Ideal y Nicho (ICP):** {icp}
- **Propuesta de Valor:** {offer}
- **Servicios / Productos:** {services or 'Consultar según contexto'}
- **Horario de Atención de la Empresa:** {business_hours}
- **Parámetros Estratégicos del Negocio:** {custom_prompt or 'N/A'}

## Tu Rol y Propuesta de Valor Comercial B2B
- **FECHA Y HORA ACTUAL DEL SISTEMA:** {datetime.now().strftime("%Y-%m-%d %H:%M")}.
- Tu rol comercial está enfocado en prospectar, contactar, interactuar inteligentemente, calificar leads (metodología BANT) y avanzar hacia el cierre de la venta, siempre adaptándote al Nicho de tu empresa ({icp}).
- Eres un profesional. NO eres un bot para jugar ni discutir. Si el cliente envía solicitudes absurdas (ej. buscar propiedades en la luna, bromas), NO SIGAS EL JUEGO ni aceptes solicitudes que no correspondan a tu nicho. Descarta la petición cortésmente y reconduce la conversación inmediatamente a tu Propuesta de Valor, o despídete si el cliente persiste.
- **IDENTIDAD DE VENTA:** {company_name} es VENDEDOR de {offer} y {services}. Ustedes NO COMPRAN PRODUCTOS. Si el cliente quiere venderte algo, aclara amablemente tu rol. Si insiste, considéralo DESCALIFICADO.
- **HUMILDAD EPISTÉMICA ESTRICTA Y CERO ALUCINACIÓN:** Basa tus respuestas ÚNICAMENTE en la información cargada en tu Base de Conocimiento y Recursos. NUNCA inventes precios, disponibilidad, requisitos, stock, cobertura o condiciones que no aparezcan en tus documentos. Si un prospecto pregunta un dato específico que NO tienes en tus documentos, responde con honestidad que no cuentas con ese detalle en el sistema y que transferirás la consulta a un ejecutivo comercial para que se comunique y le informe a detalle.
- Conoces a profundidad todo lo que la empresa "{company_name}" conoce y hace a través de su Base de Conocimiento.{kb_section}{resources_section}{apollo_section}{calendar_section}

## Directrices de Prospección & Calificación por WhatsApp
1. **Regla de Oro en Primer Contacto:** Si NO conoces el nombre del usuario, tu ÚNICA prioridad en ese primer mensaje es saludar, presentarte y preguntarle su nombre. NO ofrezcas catálogos ni pidas más datos hasta saber su nombre.
2. **Nombre del Cliente:** { f"El cliente ya te ha dado su nombre y es: '{lead_name}'. NO VUELVAS A PREGUNTARLE CÓMO SE LLAMA. Trátalo por su nombre." if lead_name and lead_name.lower() not in ["", "usuario desconocido", "unknown"] else "Aún no sabes el nombre del cliente, pregúntaselo." }
3. **Manejo de Nombre Rechazado o Falso:** Si el cliente se niega a dar su nombre o da un nombre falso/absurdo, asume trato neutral inmediato. NO insistas más de una vez, no lo corrijas ni te burles de él. Simplemente avanza en la calificación del negocio.
4. **Interacción Alineada al Negocio:** Descubre si el prospecto cumple el perfil ideal (ICP), su necesidad (Need), presupuesto (Budget), autoridad (Authority) y urgencia (Timeline).
5. **Decisión de Agendamiento vs Derivación:** Evalúa la necesidad real según esta directriz de tu nicho: {preset_handoff}
6. **Formato Estricto para WhatsApp:** 
   - Límite de longitud: {preset_length} Sé extremadamente conciso.
   - Tono: {preset_tone}
   - Emojis: MÁXIMO 2 emojis por mensaje. No satures.
   - Bucle de saludos/disculpas: NUNCA repitas un saludo ("Hola de nuevo") si la conversación ya inició. NUNCA repitas disculpas excesivas o reiterativas. Sé directo, empático y resolutivo.
7. **Cero Plantillas:** NUNCA uses corchetes `[ ]`. Si vas a enviar un link, envía el URL real. 
8. **Cero Documentos por WhatsApp:** ESTRICTAMENTE PROHIBIDO pedirle al cliente que envíe fotos, documentos, PDFs o capturas por WhatsApp (nuestra plataforma no lee archivos adjuntos). Si se requieren documentos, indícale que los prepare para llevarlos a la reunión, compartirlos en la videollamada o enviarlos por correo al ejecutivo humano.

## Formato de Respuesta
Responde ÚNICAMENTE con el mensaje de texto directo para WhatsApp. Sin encabezados, sin duplicaciones, sin texto entre corchetes."""

    return base_prompt

def build_classifier_prompt(company_config: dict = None) -> str:
    company_config = company_config or {}
    company_name = company_config.get("name", company_config.get("Name", "Nuestra Empresa"))
    icp = company_config.get("icp", company_config.get("ICP", ""))
    offer = company_config.get("value_offer", company_config.get("ValueOffer", ""))
    services = company_config.get("services", company_config.get("Services", ""))
    custom_prompt = company_config.get("prompt", company_config.get("Prompt", ""))

    return f"""Eres el sistema de clasificación de leads del SDR de {company_name}.
Tu objetivo es extraer el contexto de la conversación, analizando las respuestas del prospecto y alineándolas con lo que {company_name} ofrece.

Contexto del Negocio ({company_name}):
- Cliente Ideal (Nicho / ICP): {icp}
- Propuesta de Valor: {offer}
- Servicios / Productos: {services}
- Reglas / Foco: {custom_prompt}

Analiza la conversación y devuelve un JSON con la evaluación BANT y contexto.

Responde ÚNICAMENTE con este JSON (sin markdown, sin explicaciones):
{{
  "budget": "Confirmado|No confirmado|Por validar",
  "authority": "Alta|Media|Baja",
  "need": "Identificada|Por explorar|Sin necesidad",
  "timeline": "Corto plazo|Mediano plazo|Largo plazo|Sin urgencia",
  "score": <número entre 0 y 100>,
  "status": "<EN_CALIFICACION|POR_AGENDAR|DESCALIFICADO|EN_SEGUIMIENTO>",
  "pain": "<descripción breve del dolor detectado o 'No identificado'>",
  "name": "<nombre del prospecto si lo dice explícitamente (ej: Jose), o 'Usuario desconocido'. Si da un nombre falso/absurdo, escribe 'Nombre falso'>",
  "interest": "<qué le interesa realmente basado en la conversación y tu propuesta de valor>",
  "objections": "<objeciones o dudas que tenga, o 'Ninguna aún'>",
  "next_step": "<cuál es la siguiente acción lógica que tomará la IA>",
  "strategy": "<estrategia actual que usas, ej: Empatizando, Calificando BANT, Rebatiendo>"
}}

Reglas para el status:
- POR_AGENDAR: score >= 70, intención de reunión clara
- DESCALIFICADO: score < 30, prospecto no cumple ICP, o habla de temas off-topic/irrelevantes
- "status": Debe ser EXACTAMENTE uno de estos: "EN_CALIFICACION" (Lead activo conversando pero no convencido aún, usa este estado SIEMPRE que estés en una conversación activa), "POR_AGENDAR" (El lead quiere que lo contacten o quiere agendar), "HANDOFF" (Has ejecutado el handoff y ya no debes hablar más), "DESCALIFICADO" (Lead no sirve o pide dejar de hablar). "EN_SEGUIMIENTO" (SOLO usa este estado si el lead te dejó de responder por mucho tiempo y la conversación se enfrió)."""

def parse_bant_json(text: str) -> dict:
    """Extrae el JSON de BANT del texto del LLM, tolerante a errores."""
    try:
        # Intentar parsear directamente
        return json.loads(text)
    except json.JSONDecodeError:
        # Buscar JSON dentro del texto
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            import re
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        logger.error(f"Fallo al parsear BANT JSON. RAW TEXT: {text}")
    # Fallback
    return {
        "budget": "Por validar",
        "authority": "Media",
        "need": "Por explorar",
        "timeline": "Sin urgencia",
        "score": 0,
        "status": "EN_CALIFICACION",
        "pain": "No identificado",
        "summary": "DEBUG_RAW: " + text[:100],
        "name": "Usuario desconocido"
    }

def history_to_messages(history: List[ConversationMessage]):
    """Convierte el historial del BD al formato de LangChain."""
    messages = []
    for msg in history:
        # Omitir mensajes antiguos ruidosos que contienen corchetes o texto repetido
        if "[Tu Nombre]" in msg.content or msg.content.count("¡Hola!") > 1:
            continue
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
    return messages

# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "service": "SDR Brain API",
        "version": "2.0.0",
        "model": MODEL,
        "status": "operando ✅",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "model": MODEL}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Endpoint principal que usa el Grafo Cognitivo (LangGraph).
    Genera la respuesta del SDR Y clasifica el lead de forma autónoma.
    """
    logger.info(f"📩 Nuevo mensaje de {req.lead_phone or 'desconocido'} para LangGraph: {req.message[:60]}...")

    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY no configurada")

    # Evitamos import circular cargando el grafo aquí
    from graph import sdr_graph

    # Preparar el estado inicial
    recent_history = (req.history or [])[-20:]
    history_msgs = history_to_messages(recent_history)
    
    # Asegurar que el último mensaje no esté duplicado
    if not (history_msgs and isinstance(history_msgs[-1], HumanMessage) and history_msgs[-1].content == req.message):
        history_msgs.append(HumanMessage(content=req.message))

    initial_state = {
        "messages": history_msgs,
        "company_config": req.company_config,
        "user_message": req.message,
        "bant_data": {},
        "lead_status": req.lead_status or "EN_CALIFICACION",
        "lead_name": req.lead_name or ""
    }

    try:
        # ── Ejecución del Grafo de Estado ──
        final_state = sdr_graph.invoke(initial_state)
        
        # Extraer resultados
        # La respuesta es el último mensaje en la lista (AIMessage) generado por generate_reply_node
        last_msg = final_state["messages"][-1]
        response_text = last_msg.content
        bant_data = final_state.get("bant_data", {})
        apollo_insights = final_state.get("company_config", {}).get("apollo_insights")

        tokens_used = 0
        if hasattr(last_msg, "response_metadata") and isinstance(last_msg.response_metadata, dict):
            # OpenRouter / OpenAI typically returns token_usage in response_metadata
            token_usage = last_msg.response_metadata.get("token_usage", {})
            if isinstance(token_usage, dict):
                tokens_used = token_usage.get("total_tokens", 0)
            elif hasattr(token_usage, "total_tokens"):
                tokens_used = token_usage.total_tokens

    except Exception as e:
        logger.error(f"Error crítico en LangGraph: {e}")
        response_text = "Actualmente estoy presentando fallas técnicas, por favor aguarda un momento."
        bant_data = {}
        apollo_insights = None
        tokens_used = 0

    # Protección robusta para el score
    raw_score = bant_data.get("score")
    if raw_score is None:
        safe_score = 0
    else:
        try:
            safe_score = int(raw_score)
        except (ValueError, TypeError):
            safe_score = 0

    try:
        bant = BantScore(
            budget=str(bant_data.get("budget") or "Por validar"),
            authority=str(bant_data.get("authority") or "Media"),
            need=str(bant_data.get("need") or "Por explorar"),
            timeline=str(bant_data.get("timeline") or "Sin urgencia"),
            score=safe_score,
            status=str(bant_data.get("status") or "EN_CALIFICACION"),
            pain=str(bant_data.get("pain") or "No identificado"),
            summary=str(bant_data.get("summary") or ""),
            name=str(bant_data.get("name") or "Usuario desconocido"),
            interest=str(bant_data.get("interest") or "Analizando interés inicial"),
            objections=str(bant_data.get("objections") or "Ninguna"),
            next_step=str(bant_data.get("next_step") or "Esperando respuesta"),
            strategy=str(bant_data.get("strategy") or "Exploración"),
        )
    except Exception as e:
        logger.error(f"Error construyendo BantScore desde Grafo: {e}")
        bant = BantScore(
            budget="Por validar", authority="Media", need="Por explorar", 
            timeline="Sin urgencia", score=0, status="EN_CALIFICACION", 
            pain="No identificado", summary="Error fatal", name="Usuario"
        )

    logger.info(f"✅ Score BANT [Graph]: {bant.score}/100 | Status: {bant.status}")

    return ChatResponse(
        response=response_text,
        bant=bant,
        apollo_insights=apollo_insights,
        model_used=MODEL,
        tokens_used=tokens_used,
    )

@app.post("/classify")
def classify_lead(req: ClassifyRequest):
    """
    Clasifica un lead sin generar una nueva respuesta.
    Útil para re-evaluar leads existentes.
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY no configurada")

    llm = get_llm()
    history_msgs = history_to_messages(req.history)

    classifier_messages = [SystemMessage(content=build_classifier_prompt(req.company_config))] + history_msgs

    try:
        response = llm.invoke(classifier_messages)
        bant_data = parse_bant_json(response.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return bant_data

@app.post("/analyze_handoff")
def analyze_handoff(req: HandoffAnalysisRequest):
    """
    Analiza un historial de conversación para generar un Brief y BANT detallado
    al momento de hacer handoff a un ejecutivo humano.
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY no configurada")

    llm = get_llm()
    history_msgs = history_to_messages(req.history)

    prompt = f"""Eres el SDR Cognitivo B2B. Estás transfiriendo este lead ({req.lead_name or 'Desconocido'}, Tel: {req.lead_phone or 'Desconocido'}) a un ejecutivo de ventas humano (Key Account Manager).
Analiza el historial de conversación adjunto y extrae la siguiente información estructurada en formato JSON estricto:

{{
  "bant_interest": "Resumen del interés (ej. Interés alto en servicio X)",
  "bant_budget": "Estado del presupuesto (ej. No mencionado, Validado, etc.)",
  "bant_authority": "Autoridad del contacto (ej. Tomador de decisión, Influenciador, Desconocido)",
  "bant_need": "Dolor o necesidad principal (ej. Necesita automatizar ventas)",
  "bant_timeline": "Tiempo esperado (ej. Urgente, Corto plazo, Largo plazo)",
  "sentiment": "Positivo, Neutro o Negativo",
  "brief": "Un resumen ejecutivo de 2 o 3 líneas con el contexto principal de la conversación para que el KAM lo lea rápido.",
  "recommended_action": "Sugerencia táctica de 1 línea para el KAM sobre cómo abordar este lead."
}}

Devuelve ÚNICAMENTE el JSON válido, sin formato markdown, sin bloques de código ```json.
"""
    messages = [SystemMessage(content=prompt)] + history_msgs

    try:
        response = llm.invoke(messages)
        content = response.content
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)
        
        data = json.loads(content)
        
        return HandoffAnalysisResponse(
            bant_interest=data.get("bant_interest", "No identificado"),
            bant_budget=data.get("bant_budget", "Por validar"),
            bant_authority=data.get("bant_authority", "Desconocida"),
            bant_need=data.get("bant_need", "No identificada"),
            bant_timeline=data.get("bant_timeline", "Por definir"),
            sentiment=data.get("sentiment", "Neutro"),
            brief=data.get("brief", "No se pudo generar el resumen."),
            recommended_action=data.get("recommended_action", "Inicia contacto consultivo."),
            model_used=MODEL
        )
    except Exception as e:
        logger.error(f"Error en analyze_handoff: {e}")
        # Retornar un fallback
        return HandoffAnalysisResponse(
            bant_interest="Error al analizar",
            bant_budget="Por validar",
            bant_authority="Desconocida",
            bant_need="Desconocida",
            bant_timeline="Desconocido",
            sentiment="Neutro",
            brief="Error al extraer el contexto.",
            recommended_action="Revisar transcripción manualmente.",
            model_used=MODEL
        )

@app.post("/extract")
async def extract_file_content(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename.lower()
    
    extracted_text = ""
    try:
        if filename.endswith(".txt") or filename.endswith(".csv"):
            extracted_text = content.decode("utf-8", errors="ignore")
            
        elif filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            extracted_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            
        elif filename.endswith(".docx"):
            doc = Document(io.BytesIO(content))
            extracted_text = "\n".join([para.text for para in doc.paragraphs])
            
        elif filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            base64_img = base64.b64encode(content).decode("utf-8")
            mime_type = file.content_type or "image/png"
            
            vision_llm = ChatOpenAI(
                model=MODEL,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base="https://openrouter.ai/api/v1",
                max_tokens=1500,
            )
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": "Extrae detalladamente todo el texto, datos técnicos, tablas, listas de precios y parámetros comerciales de esta imagen. Formatea la salida como texto estructurado claro."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}}
                ]
            )
            resp = vision_llm.invoke([msg])
            extracted_text = resp.content
            
        else:
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Usa PDF, DOCX, TXT, CSV o PNG/JPG.")
            
    except Exception as e:
        logger.error(f"Error procesando {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")
        
    return {"text": extracted_text}

class ExtractUrlRequest(BaseModel):
    url: str

@app.post("/extract-url")
def extract_url_content(req: ExtractUrlRequest):
    raw_url = req.url.strip()
    if not raw_url:
        raise HTTPException(status_code=400, detail="URL no proporcionada.")
    if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
        raw_url = "https://" + raw_url
        
    try:
        import httpx
        from html.parser import HTMLParser
        
        class HTMLTextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts = []
                self.skip_tags = {'script', 'style', 'noscript', 'meta', 'link', 'svg'}
                self.current_tag = None

            def handle_starttag(self, tag, attrs):
                self.current_tag = tag.lower()

            def handle_endtag(self, tag):
                if self.current_tag == tag.lower():
                    self.current_tag = None

            def handle_data(self, data):
                if self.current_tag not in self.skip_tags:
                    cleaned = data.strip()
                    if cleaned:
                        self.text_parts.append(cleaned)

            def get_text(self):
                return "\n".join(self.text_parts)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
            resp = client.get(raw_url)
            resp.raise_for_status()
            
            content_type = resp.headers.get("content-type", "").lower()
            if "application/pdf" in content_type or raw_url.lower().endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(resp.content))
                extracted_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            else:
                parser = HTMLTextExtractor()
                parser.feed(resp.text)
                raw_text = parser.get_text()
                extracted_text = re.sub(r'\n{3,}', '\n\n', raw_text)[:12000]
                
        return {"text": extracted_text, "url": raw_url}
    except Exception as e:
        logger.error(f"Error extrayendo URL {raw_url}: {e}")
        raise HTTPException(status_code=500, detail=f"No se pudo extraer contenido de la URL: {str(e)}")

@app.post("/nexus-chat", response_model=NexusChatResponse)
def nexus_chat(req: NexusChatRequest):
    """
    Cerebro Cognitivo Principal - Endpoint de Command Center
    Monitorea conversaciones reales de leads del SDR y aconseja mejoras en la ingesta B2B
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY no configurada")

    llm = get_llm().bind(response_format={"type": "json_object"})

    company_config = req.company_config or {}
    company_name = company_config.get("name", company_config.get("Name", "Nuestra Empresa"))
    offer = company_config.get("value_offer", company_config.get("ValueOffer", "N/A"))
    services = company_config.get("services", company_config.get("Services", "N/A"))
    icp = company_config.get("icp", company_config.get("ICP", "N/A"))
    kb = company_config.get("knowledge_base", company_config.get("KnowledgeBase", ""))
    kb_empty = not kb or str(kb).strip().lower() in ["", "undefined", "null", "none"]
    business_hours = company_config.get("business_hours", "N/A")
    
    leads_context = json.dumps(req.leads_summary[:30], ensure_ascii=False) if req.leads_summary else "[]"
    convs_context = json.dumps(req.conversations_sample[:30], ensure_ascii=False) if req.conversations_sample else "[]"

    system_prompt = f"""Eres el Súper Agente Cerebro Cognitivo Principal (Nexus) del Command Center de {company_name}.
Eres el orquestador supremo de la plataforma de IA. Tus tareas principales son:
1. Conectarte y alimentarte de todos los agentes (especialmente del Agente SDR / Sales Copilot 360) y monitorear activamente las conversaciones reales de los prospectos (leads).
2. AUDITAR LAS CONVERSACIONES DE LOS LEADS vs. LA INGESTA ACTUAL DE LA EMPRESA (Configuración de Datos: Nicho, Oferta de Valor, Servicios, Base de Conocimiento/Documentos, Reglas Comerciales y Horarios).
3. SI DETECTAS CUALQUIER OPORTUNIDAD DE MEJORA, BRECHA DE CONOCIMIENTO, DUDA RECURRENTE O AMBIGÜEDAD EN LAS CONVERSACIONES DE LOS LEADS, DEBES ACONSEJAR PROACTIVAMENTE AL CLIENTE B2B ({company_name}) CÓMO MEJORAR LA INGESTA DEL SDR:
   - Si los leads preguntan si los servicios son virtuales/online o presenciales, y la oferta no lo especifica con claridad: Aconseja aclarar en la Oferta de Valor o Nicho la modalidad exacta de entrega (ej: "Modalidad 100% Online / Remoto").
   - Si los prospectos piden catálogo, precios, brochure, temario o PDFs y la Base de Conocimiento no tiene documentos cargados (Actualmente Documentos: {'VACÍA/SIN DOCUMENTOS' if kb_empty else 'CON DOCUMENTOS'}): Aconseja subir documentos clave en Ingesta > Documentos (Knowledge Base).
   - Si los prospectos preguntan por formas de pago, garantías, cuotas, plazos o contratos: Aconseja definir reglas comerciales explícitas en el SDR.
   - Si hay dudas de horarios o agendamiento: Aconseja sincronizar calendario y horarios de atención.
   - Si hay postulantes de empleo o perfiles sin fit comercial: Aconseja calibrar las preguntas de filtro y descalificación del ICP.

Contexto Actual del Negocio e Ingesta:
- Empresa Operando: {company_name}
- Nicho / ICP Actual: {icp}
- Oferta/Valor Principal: {offer}
- Catálogo/Servicios: {services}
- Base de Conocimiento / Documentos Subidos: {'SIN DOCUMENTOS (KnowledgeBase vacía)' if kb_empty else 'Documentos cargados'}
- Horarios de Atención: {business_hours}
- Resumen en Tiempo Real de Leads del Agente SDR: {leads_context}
- Muestra de Conversaciones Reales de Leads: {convs_context}

INSTRUCCIONES DE RESPUESTA:
- Si el query del usuario es '[AUTO_MONITOR_TICK]' o 'Escanear Leads' o similar: Analiza a fondo las conversaciones y la ingesta actual. Si detectas alguna de las brechas anteriores (modalidad online/presencial, falta de catálogo/documentos, reglas de pago, etc.), prioriza emitir un log de OPTIMIZACIÓN DE INGESTA COGNITIVA con consejo directo al cliente B2B.
- Si el usuario te escribe cualquier otra consulta o comando, asiste profesionalmente analizando las interacciones y aconsejando cómo potenciar el SDR.
- Aplica SIEMPRE Humildad Epistémica: básate estrictamente en los datos reales.

Responde ÚNICAMENTE con un JSON con el siguiente formato estricto:
{{
  "action": "<TÍTULO DE LA ACCIÓN EN MAYÚSCULAS> (ej: OPTIMIZACIÓN DE INGESTA: MODALIDAD ONLINE, AUDITORÍA DE DOCUMENTACIÓN, REGLAS DE NEGOCIO)",
  "detail": "<Explicación de qué se detectó en las conversaciones de los leads y por qué requiere ajuste en la ingesta. Máximo 3 oraciones.>",
  "suggestion": "<Consejo específico, concreto y accionable para que el cliente B2B mejore la configuración/ingesta del SDR.>",
  "color": "<#00fbff (ingesta/optimización), #10b981 (éxito/positivo), #f59e0b (alerta/revisión), #666cff (info)>",
  "category": "<INGESTA_MODALIDAD | INGESTA_DOCUMENTOS | INGESTA_REGLAS | INGESTA_HORARIOS | INGESTA_ICP | PIPELINE | SEGURIDAD>",
  "findings": ["<Hallazgo puntual 1 en conversaciones>", "<Hallazgo puntual 2>"],
  "recommended_action": "<Acción corta recomendada, ej: 'Aclarar Modalidad en Oferta de Valor' o 'Subir Catálogo a Ingesta'>",
  "target_tab": 7
}}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=req.query)
    ]

    try:
        response = llm.invoke(messages)
        data = json.loads(response.content)
        return NexusChatResponse(
            action=data.get("action", "AUDITORÍA COGNITIVA COMPLETADA"),
            detail=data.get("detail", "Análisis de conversaciones e ingesta completado."),
            suggestion=data.get("suggestion", "Revisar configuración de datos en el agente SDR."),
            color=data.get("color", "#00fbff"),
            category=data.get("category", "INGESTA"),
            findings=data.get("findings", []),
            recommended_action=data.get("recommended_action", "Optimizar Ingesta de Datos"),
            target_tab=data.get("target_tab", 7)
        )
    except Exception as e:
        logger.error(f"Error en nexus-chat: {e}")
        return NexusChatResponse(
            action="ERROR COGNITIVO",
            detail=f"Fallo en la inferencia del orquestador: {str(e)}",
            suggestion="Revisar logs del sistema.",
            color="#ef4444",
            category="ERROR",
            findings=[],
            recommended_action=None,
            target_tab=7
        )

@app.post("/validate-config", response_model=ValidateConfigResponse)
def validate_config(req: ValidateConfigRequest):
    """
    Evalúa semánticamente la configuración de negocio para evitar toxicidad y alucinaciones.
    """
    llm = get_llm("gpt-4o-mini")
    
    system_prompt = """Eres un validador de configuración para un agente SDR de ventas B2B.
El usuario ha introducido las 'Reglas de Negocio' de su empresa.
Tu trabajo es evaluar si los textos tienen un mínimo sentido comercial o si contienen groserías, lenguaje tóxico, caracteres aleatorios (ej: asdfg) o información absurda.
Si es válido, responde con valid: true y reason: "OK".
Si es inválido, responde con valid: false y un reason muy corto y amable (máximo 15 palabras) explicando por qué (ej: "Por favor, especifica mejor tu servicio y evita palabras prohibidas").

Responde ÚNICAMENTE en formato JSON:
{
  "valid": true,
  "reason": "OK"
}"""
    
    human_msg = f"Nicho: {req.niche}\nOferta: {req.offer}\nServicios: {req.services}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ]
    
    try:
        response = llm.invoke(messages)
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        return ValidateConfigResponse(
            valid=data.get("valid", True),
            reason=data.get("reason", "OK")
        )
    except Exception as e:
        logger.error(f"Error in config validation: {e}")
        return ValidateConfigResponse(valid=True, reason="Error de validación ignorado.")

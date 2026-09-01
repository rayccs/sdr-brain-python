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
# Preferimos anthropic/claude-3.5-sonnet por ser el mejor modelo lógico/comercial en OpenRouter
MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-5")

def get_llm(model_override: Optional[str] = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_override or MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.6,
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
    model_used: str

class NexusChatRequest(BaseModel):
    query: str
    company_config: Optional[dict] = None
    leads_summary: Optional[List[dict]] = []

class NexusChatResponse(BaseModel):
    action: str
    detail: str
    suggestion: str
    color: str

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
def build_system_prompt(company_config: Optional[dict], user_query: str = "") -> str:
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

    base_prompt = f"""Eres {agent_name}, un asesor comercial humano altamente capacitado para la empresa "{company_name}". 
IMPORTANTE: Nunca menciones que eres una IA, un bot o un "SDR Cognitivo". Comórtate siempre como un humano natural, aplicando los principios de Humildad Epistémica (reconoce cordialmente si no sabes algo en lugar de inventarlo o alucinar) y Entropía Negativa (aporta siempre estructura, claridad y valor a la conversación del cliente en lugar de generar ruido o respuestas vagas). Usa RAG estrictamente basado en tu Base de Conocimiento.

## Tu Rol y Propuesta de Valor Comercial B2B
- Tu rol comercial está enfocado en prospectar, contactar, interactuar inteligentemente, calificar leads (metodología BANT) y agendar reuniones para los ejecutivos de cuenta, sin realizar el cierre final.
- Conoces a profundidad todo lo que la empresa "{company_name}" conoce y hace a través de su Base de Conocimiento.{kb_section}{resources_section}

## Conocimiento de Negocio del Cerebro de Ventas ({company_name})
- **Cliente Ideal (ICP):** {icp}
- **Propuesta de Valor:** {offer}
- **Servicios / Productos:** {services or 'Consultar según contexto'}
- **Parámetros Estratégicos del Negocio:** {custom_prompt or 'N/A'}

## Directrices de Prospección & Calificación por WhatsApp
1. **Regla de Oro en Primer Contacto:** Si el lead inicia la conversación y su nombre es "Usuario desconocido" o no lo sabes, tu **ÚNICA** prioridad en ese primer mensaje es saludar, presentarte y **preguntarle su nombre**. NO pidas su número de WhatsApp (porque ya están hablando por ahí). NO ofrezcas catálogos ni hagas preguntas de negocio hasta que te diga su nombre.
2. **Reconducción de Conversaciones (Off-Topic):** Si el lead pregunta o habla sobre temas que NO tienen absolutamente nada que ver con los servicios o productos de {company_name}, DEBES reconducir amablemente la conversación hacia lo que sí ofrecemos. Si el lead insiste en temas fuera de lugar o no muestra interés comercial, despídete cortésmente y da por cerrada la conversación.
3. **Interacción Alineada al Negocio:** Solo después de saber su nombre, puedes entregar la información solicitada o hacer preguntas inteligentes para descubrir si cumple el perfil ideal (ICP), su necesidad (Need), presupuesto (Budget), autoridad (Authority) y urgencia (Timeline).
4. **Agendamiento con KAM:** Cuando detectes interés claro o alta coincidencia con el ICP, propón agendar una breve llamada/demo con un ejecutivo de cuenta de {company_name}.
5. **Recursos y Catálogos (PROACTIVIDAD):** Si en tu Base de Conocimiento tienes enlaces a "RECURSOS PARA COMPARTIR" (Catálogos, Dossiers, etc.), debes ser proactivo. Cuando el cliente pregunte por productos, precios o pida más información, dile que tienes un catálogo a la mano y pregúntale si quiere que se lo envíes. Si te dice que sí o de entrada pide el catálogo, envíale EXACTAMENTE el Link Público (URL real) que aparece en tus recursos. **PROHIBIDO inventar links o usar corchetes (ej. "[link al catálogo]")**.
6. **Formato Estricto para WhatsApp:** Responde con UN SOLO mensaje corto, natural y empático (máximo 2 a 3 oraciones en un solo párrafo, menos de 50 palabras).
7. **Cero Plantillas e Identidad 100% Humana:** NUNCA uses corchetes `[ ]`. Si vas a enviar un link, envía el URL real. NUNCA digas que eres un "SDR Cognitivo" ni una IA; si te preguntan, responde siempre desde tu identidad humana como asesor de {company_name}.
8. **Emoticones Contextuales:** Incluye siempre emoticones sutiles, profesionales y acordes al contexto del mensaje (1 a 3 por respuesta), para hacer la conversación de WhatsApp más dinámica y cercana.

## Formato de Respuesta
Responde ÚNICAMENTE con el mensaje de texto directo para WhatsApp. Sin encabezados, sin duplicaciones, sin texto entre corchetes."""

    return base_prompt

def build_classifier_prompt(company_config: dict = None) -> str:
    company_config = company_config or {}
    company_name = company_config.get("name", company_config.get("Name", "Nuestra Empresa"))
    offer = company_config.get("value_offer", company_config.get("ValueOffer", ""))
    services = company_config.get("services", company_config.get("Services", ""))
    custom_prompt = company_config.get("prompt", company_config.get("Prompt", ""))

    return f"""Eres el sistema de clasificación de leads del SDR de {company_name}.
Tu objetivo es extraer el contexto de la conversación, analizando las respuestas del prospecto y alineándolas con lo que {company_name} ofrece.

Contexto del Negocio ({company_name}):
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
  "name": "<nombre del prospecto si lo dice explícitamente (ej: Jose), o 'Usuario desconocido'>",
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
        if "[Tu Nombre]" in msg.content or msg.content.count("¡Hola!") > 1 or len(msg.content) > 250:
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
    Endpoint principal que N8N llama cuando llega un mensaje de WhatsApp.
    Genera la respuesta del SDR Y clasifica el lead en el mismo ciclo.
    """
    logger.info(f"📩 Nuevo mensaje de {req.lead_phone or 'desconocido'}: {req.message[:60]}...")

    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY no configurada")

    llm = get_llm()

    # ── Paso 1: Generar respuesta conversacional ──────────────────
    system_prompt = build_system_prompt(req.company_config, req.message)
    # Limitar el historial a máximo los últimos 6 mensajes para evitar contaminación
    recent_history = (req.history or [])[-6:]
    history_msgs = history_to_messages(recent_history)

    sdr_messages = [SystemMessage(content=system_prompt)] + history_msgs
    if not (history_msgs and isinstance(history_msgs[-1], HumanMessage) and history_msgs[-1].content == req.message):
        sdr_messages.append(HumanMessage(content=req.message))

    try:
        sdr_response = llm.invoke(sdr_messages)
        response_text = sdr_response.content
    except Exception as e:
        logger.error(f"Error generando respuesta SDR: {e}")
        raise HTTPException(status_code=400, detail=f"Error del LLM: {str(e)}")

    # ── Paso 2: Clasificar el lead con el historial completo ──────
    full_history = history_to_messages(req.history or []) + [
        HumanMessage(content=req.message),
        AIMessage(content=response_text),
    ]

    classifier_messages = [SystemMessage(content=build_classifier_prompt(req.company_config))] + full_history + [
        HumanMessage(content="Clasifica el lead ahora basado en esta conversación y entrega el JSON solicitado.")
    ]

    try:
        classifier_llm = get_llm().bind(response_format={"type": "json_object"})
        bant_response = classifier_llm.invoke(classifier_messages)
        bant_data = parse_bant_json(bant_response.content)
    except Exception as e:
        logger.warning(f"Error clasificando lead: {e}")
        bant_data = {
            "budget": "Por validar", "authority": "Media", "need": "Por explorar",
            "timeline": "Sin urgencia", "score": 10, "status": "EN_CALIFICACION",
            "pain": "No identificado", "summary": "Error de clasificación"
        }

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
        logger.error(f"Error construyendo BantScore: {e}")
        # Fallback extremo seguro
        bant = BantScore(
            budget="Por validar", authority="Media", need="Por explorar", 
            timeline="Sin urgencia", score=0, status="EN_CALIFICACION", 
            pain="No identificado", summary="Error fatal", name="Usuario"
        )

    logger.info(f"✅ Score BANT: {bant.score}/100 | Status: {bant.status}")

    return ChatResponse(
        response=response_text,
        bant=bant,
        model_used=MODEL,
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
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "").strip()
        
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

@app.post("/nexus-chat", response_model=NexusChatResponse)
def nexus_chat(req: NexusChatRequest):
    """
    Cerebro Cognitivo Principal - Endpoint de Command Center
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY no configurada")

    llm = get_llm().bind(response_format={"type": "json_object"})

    company_config = req.company_config or {}
    company_name = company_config.get("name", company_config.get("Name", "Nuestra Empresa"))
    offer = company_config.get("value_offer", company_config.get("ValueOffer", "N/A"))
    
    leads_context = json.dumps(req.leads_summary[:50]) # limit context

    system_prompt = f"""Eres el Cerebro Cognitivo Principal (Nexus) del Command Center de {company_name}.
Eres un orquestador de IA avanzado que analiza el tráfico de leads en tiempo real y la configuración del negocio para dar respuestas estratégicas de alto nivel directivas.

Contexto del Negocio:
- Empresa: {company_name}
- Oferta/Valor: {offer}
- Leads Actuales (Resumen): {leads_context}

El usuario ha ejecutado un comando o te ha hecho una pregunta. Analiza los datos de los leads, la situación, y responde con una acción estratégica.
Responde ÚNICAMENTE con un JSON con el siguiente formato:
{{
  "action": "<TÍTULO DE LA ACCIÓN EN MAYÚSCULAS> (ej: ANÁLISIS DE ROI EJECUTADO: ...)",
  "detail": "<Detalle analítico basado en los leads y negocio (ej: Se detectan 3 leads descartados y 2 agendados...)>",
  "suggestion": "<Una recomendación o siguiente paso accionable>",
  "color": "<código hex de color asociado al sentimiento de la acción, ej: #10b981 (éxito), #f59e0b (alerta), #666cff (informativo)>"
}}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=req.query)
    ]

    try:
        response = llm.invoke(messages)
        data = json.loads(response.content)
        return NexusChatResponse(
            action=data.get("action", "COMANDO PROCESADO"),
            detail=data.get("detail", "Análisis completado."),
            suggestion=data.get("suggestion", "N/A"),
            color=data.get("color", "#666cff")
        )
    except Exception as e:
        logger.error(f"Error en nexus-chat: {e}")
        return NexusChatResponse(
            action="ERROR COGNITIVO",
            detail=f"Fallo en la inferencia del orquestador: {str(e)}",
            suggestion="Revisar logs del sistema.",
            color="#ef4444"
        )

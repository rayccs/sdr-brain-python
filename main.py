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
import json
import logging
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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

# ──────────────────────────────────────────────────────────────────
# LLM — usa OpenRouter con modelo configurable
# ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Preferimos google/gemini-2.5-flash (muy bueno y barato).
# Si no hay crédito, cae a llama-3-8b:free
MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")

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

class ChatResponse(BaseModel):
    response: str
    bant: BantScore
    model_used: str

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def build_system_prompt(company_config: Optional[dict]) -> str:
    """
    Construye el system prompt del agente SDR basado en la configuración
    de la empresa cliente. Si no hay config, usa valores genéricos.
    """
    if company_config:
        icp = company_config.get("icp", company_config.get("ICP", "empresas B2B"))
        offer = company_config.get("value_offer", company_config.get("ValueOffer", "soluciones de automatización"))
        services = company_config.get("services", company_config.get("Services", ""))
        custom_prompt = company_config.get("prompt", company_config.get("Prompt", ""))
    else:
        icp = "empresas B2B medianas y grandes"
        offer = "soluciones de automatización e inteligencia artificial"
        services = "Agentes IA, automatizaciones de ventas, CRM inteligente"
        custom_prompt = ""

    base_prompt = f"""Eres un SDR (Sales Development Representative) cognitivo B2B de élite. 
Tu misión es calificar prospectos de manera natural y empática a través de WhatsApp.

## Tu Perfil Profesional
- Eres experto en ventas consultivas B2B
- Aplicas la metodología BANT (Budget, Authority, Need, Timeline) de forma natural
- Haces preguntas estratégicas para descubrir el dolor del cliente sin ser invasivo
- Tu objetivo: identificar leads de calidad y agendar reuniones con el equipo KAM

## Empresa que Representas
- **Cliente Ideal (ICP):** {icp}
- **Propuesta de Valor:** {offer}
- **Servicios/Productos:** {services or 'Consultar según contexto'}

## Instrucciones de Comportamiento
1. Responde SIEMPRE en español (a menos que el prospecto escriba en otro idioma)
2. Mensajes CORTOS (máximo 3 párrafos), directos y en tono conversacional
3. Haz UNA sola pregunta estratégica por mensaje
4. Cuando detectes alta intención de compra, propón agendar una llamada/demo
5. No reveles que eres una IA a menos que te pregunten directamente
6. Si el prospecto NO es cliente ideal, descalifica amablemente
7. NUNCA escribas corchetes ni marcadores de posición como "[Tu Nombre]", "[Mi Nombre]" o "[Empresa]". Preséntate como un consultor del equipo.

## Guía de Calificación BANT
- **Budget:** Preguntar sobre presupuesto disponible o inversión esperada
- **Authority:** Identificar si es el tomador de decisión
- **Need:** Descubrir el dolor específico que tienen
- **Timeline:** Cuándo necesitan resolver el problema

{custom_prompt}

## Formato de Respuesta
Responde SOLO el texto del mensaje de WhatsApp. Sin emojis de cabecera innecesarios.
Usa emojis con moderación para hacer el mensaje más cercano."""

    return base_prompt

def build_classifier_prompt() -> str:
    return """Eres un sistema de clasificación de leads B2B. 
Analiza la conversación y devuelve un JSON con la evaluación BANT.

Responde ÚNICAMENTE con este JSON (sin markdown, sin explicaciones):
{
  "budget": "Confirmado|No confirmado|Por validar",
  "authority": "Alta|Media|Baja",
  "need": "Identificada|Por explorar|Sin necesidad",
  "timeline": "Corto plazo|Mediano plazo|Largo plazo|Sin urgencia",
  "score": <número entre 0 y 100>,
  "status": "<EN_CALIFICACION|POR_AGENDAR|DESCALIFICADO|EN_SEGUIMIENTO>",
  "pain": "<descripción breve del dolor detectado o 'No identificado'>",
  "summary": "<resumen de 1 oración del estado del lead>"
}

Reglas para el status:
- POR_AGENDAR: score >= 70, intención de reunión clara
- DESCALIFICADO: score < 30 o prospecto no cumple ICP
- EN_SEGUIMIENTO: score 30-69, interesado pero no listo
- EN_CALIFICACION: conversación muy nueva, aún no hay suficiente info"""

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
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    # Fallback
    return {
        "budget": "Por validar",
        "authority": "Media",
        "need": "Por explorar",
        "timeline": "Sin urgencia",
        "score": 0,
        "status": "EN_CALIFICACION",
        "pain": "No identificado",
        "summary": "Iniciando calificación"
    }

def history_to_messages(history: List[ConversationMessage]):
    """Convierte el historial del BD al formato de LangChain."""
    messages = []
    for msg in history:
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
    system_prompt = build_system_prompt(req.company_config)
    history_msgs = history_to_messages(req.history or [])

    sdr_messages = [SystemMessage(content=system_prompt)] + history_msgs
    if not (history_msgs and isinstance(history_msgs[-1], HumanMessage) and history_msgs[-1].content == req.message):
        sdr_messages.append(HumanMessage(content=req.message))

    try:
        sdr_response = llm.invoke(sdr_messages)
        response_text = sdr_response.content
    except Exception as e:
        logger.error(f"Error generando respuesta SDR: {e}")
        raise HTTPException(status_code=500, detail=f"Error del LLM: {str(e)}")

    # ── Paso 2: Clasificar el lead con el historial completo ──────
    full_history = history_to_messages(req.history or []) + [
        HumanMessage(content=req.message),
        AIMessage(content=response_text),
    ]

    classifier_messages = [SystemMessage(content=build_classifier_prompt())] + full_history

    try:
        bant_response = llm.invoke(classifier_messages)
        bant_data = parse_bant_json(bant_response.content)
    except Exception as e:
        logger.warning(f"Error clasificando lead: {e}")
        bant_data = {
            "budget": "Por validar", "authority": "Media", "need": "Por explorar",
            "timeline": "Sin urgencia", "score": 10, "status": "EN_CALIFICACION",
            "pain": "No identificado", "summary": "Error de clasificación"
        }

    bant = BantScore(
        budget=bant_data.get("budget", "Por validar"),
        authority=bant_data.get("authority", "Media"),
        need=bant_data.get("need", "Por explorar"),
        timeline=bant_data.get("timeline", "Sin urgencia"),
        score=int(bant_data.get("score", 0)),
        status=bant_data.get("status", "EN_CALIFICACION"),
        pain=bant_data.get("pain", "No identificado"),
        summary=bant_data.get("summary", ""),
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

    classifier_messages = [SystemMessage(content=build_classifier_prompt())] + history_msgs

    try:
        response = llm.invoke(classifier_messages)
        bant_data = parse_bant_json(response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return bant_data

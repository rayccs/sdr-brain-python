import os
import json
import logging
import requests
import re
from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

# Importamos langgraph
from langgraph.graph import StateGraph, START, END

# Importar dependencias de main.py
from main import get_llm, get_gemini_llm, build_system_prompt, build_classifier_prompt, parse_bant_json
from langchain_core.tools import tool
import calendar_tools

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# 1. Definición del Estado
# ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: List[BaseMessage]        # Historial + Mensaje nuevo + Respuesta generada
    company_config: Dict[str, Any]     # Configuración del cliente (ICP, Knowledge Base)
    user_message: str                  # El mensaje original del lead
    bant_data: Dict[str, Any]          # El resultado de la clasificación final
    lead_status: str                   # Estado del lead (ej. HANDOFF)
    lead_name: str                     # Nombre del lead

# ──────────────────────────────────────────────────────────────────
# 2. Nodos del Grafo
# ──────────────────────────────────────────────────────────────────

def extract_email(text: str) -> str:
    if not isinstance(text, str):
        return None
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def apollo_enrichment_node(state: AgentState) -> Dict[str, Any]:
    """
    Intenta enriquecer el lead usando Apollo.io si la configuración está activa y hay un email.
    """
    company_config = state.get("company_config", {})
    apollo_enabled = company_config.get("apollo_enabled", False)
    apollo_api_key = company_config.get("apollo_api_key", "")
    
    if not apollo_enabled or not apollo_api_key:
        return {}
        
    email_to_enrich = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            found_email = extract_email(msg.content)
            if found_email:
                email_to_enrich = found_email
                break
                
    if not email_to_enrich:
        return {}
        
    if "apollo_insights" in company_config:
        return {}
        
    logger.info(f"🔎 [Nodo: apollo_enrichment] - Extrayendo datos de Apollo para: {email_to_enrich}")
    
    try:
        url = "https://api.apollo.io/v1/people/match"
        headers = {"Content-Type": "application/json", "Cache-Control": "no-cache"}
        data = {"api_key": apollo_api_key, "email": email_to_enrich}
        
        response = requests.post(url, headers=headers, json=data, timeout=5)
        if response.status_code == 200:
            apollo_data = response.json()
            person = apollo_data.get("person", {})
            if person:
                org = person.get("organization", {})
                title = person.get("title", "Desconocido")
                seniority = person.get("seniority", "Desconocido")
                company = org.get("name", "Desconocida")
                industry = org.get("industry", "Desconocida")
                employees = org.get("estimated_num_employees", "Desconocido")
                
                insights = f"El prospecto es {title} ({seniority}) en la empresa {company} (Industria: {industry}, Empleados: {employees}). Usa esta información a tu favor."
                
                updated_config = dict(company_config)
                updated_config["apollo_insights"] = insights
                return {"company_config": updated_config}
    except Exception as e:
        logger.warning(f"Error o límite alcanzado en Apollo API: {e}")
        
    return {}

def generate_reply_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo encargado de leer el contexto y la memoria RAG para responder con Humildad Epistémica.
    """
    logger.info("⚡ [Nodo: generate_reply] - Generando respuesta SDR...")
    
    if state.get("lead_status") in ["HANDOFF", "POR_AGENDAR"]:
        logger.info(f"⚡ [Nodo: generate_reply] - Lead en {state.get('lead_status')}, operando en modo SHADOW (sin generar respuesta IA)...")
        # Devolvemos un mensaje vacío para que no responda al prospecto, 
        # pero permitimos que continue al clasificador para actualizar BANT.
        return {"messages": [AIMessage(content="")]}

    hermes_llm = get_llm()
    gemini_llm = get_gemini_llm()
    user_message = state["user_message"]
    company_config = state["company_config"]
    history_msgs = state["messages"]
    lead_name = state.get("lead_name", "")

    system_prompt = build_system_prompt(company_config, user_message, lead_name)
    
    # Construir mensajes (System + History + Último Mensaje Humano)
    sdr_messages = [SystemMessage(content=system_prompt)] + history_msgs

    # --- Herramientas de Calendario ---
    auto_schedule = company_config.get("auto_schedule_enabled", False)
    access_token = company_config.get("google_access_token", "")
    refresh_token = company_config.get("google_refresh_token", "")
    
    active_tools = []
    if auto_schedule and access_token:
        @tool
        def check_availability(date_str: str) -> str:
            """Revisa la disponibilidad en Google Calendar para una fecha (YYYY-MM-DD). Usa esto ANTES de agendar para sugerir horarios libres."""
            return calendar_tools.check_calendar_availability(access_token, refresh_token, date_str)
            
        @tool
        def book_appointment(start_time: str, end_time: str, summary: str, description: str, attendee_email: str = None) -> str:
            """Agenda una reunión en Google Calendar. start_time y end_time deben ser formato ISO 8601 (ej. 2023-10-25T10:00:00-03:00)."""
            return calendar_tools.book_calendar_appointment(access_token, refresh_token, start_time, end_time, summary, description, attendee_email)
            
        active_tools = [check_availability, book_appointment]
        gemini_llm = gemini_llm.bind_tools(active_tools)

    try:
        # 1. Primera pasada: Gemini decide si usar herramientas
        sdr_response = gemini_llm.invoke(sdr_messages)
        
        # Ejecutar tools si Gemini las invocó
        if active_tools and hasattr(sdr_response, "tool_calls") and sdr_response.tool_calls:
            logger.info(f"🔧 Gemini invocó tools: {sdr_response.tool_calls}")
            sdr_messages.append(sdr_response) # Añadimos el mensaje de la IA con la llamada
            
            tool_map = {t.name: t for t in active_tools}
            for tool_call in sdr_response.tool_calls:
                tool_instance = tool_map.get(tool_call["name"])
                if tool_instance:
                    tool_output = tool_instance.invoke(tool_call["args"])
                    from langchain_core.messages import ToolMessage
                    sdr_messages.append(ToolMessage(
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                        content=str(tool_output)
                    ))
                    
            # Segunda llamada: Hermes genera la respuesta final con el resultado de las tools
            sdr_response = hermes_llm.invoke(sdr_messages)
        else:
            # Si Gemini NO invocó tools, usamos a Hermes para responder
            sdr_response = hermes_llm.invoke(sdr_messages)

        # --- Nodepath Integration ---
        content = sdr_response.content
        
        # Ensure content is a string
        if isinstance(content, list):
            content = " ".join([str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content])
        elif not isinstance(content, str):
            content = str(content)
            
        urls = re.findall(r'(https?://[^\s]+)', content)
        for url in set(urls):
            if "nodepath.link" not in url:
                short_url = f"https://nodepath.link/{url.__hash__() % 100000:05x}"
                content = content.replace(url, short_url)
        
        # Solo reasignamos si era un string originalmente, o si lo convertimos a string.
        if isinstance(sdr_response.content, str):
            sdr_response.content = content
        
    except Exception as e:
        logger.error(f"Error generando respuesta SDR en LangGraph: {e}")
        sdr_response = AIMessage(content="Actualmente estoy presentando fallas técnicas, por favor aguarda un momento.")

    return {"messages": [sdr_response]}

def classify_lead_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo que lee la conversación (incluyendo la respuesta generada) y evalúa el perfil BANT.
    """
    logger.info("⚡ [Nodo: classify_lead] - Clasificando BANT...")
    
    company_config = state["company_config"]
    # state["messages"] ahora contiene la respuesta generada por generate_reply_node
    full_history = state["messages"] 

    classifier_messages = [SystemMessage(content=build_classifier_prompt(company_config))] + full_history + [
        HumanMessage(content="Clasifica el lead ahora basado en esta conversación y entrega el JSON solicitado.")
    ]

    try:
        classifier_llm = get_llm().bind(response_format={"type": "json_object"})
        bant_response = classifier_llm.invoke(classifier_messages)
        bant_data = parse_bant_json(bant_response.content)
    except Exception as e:
        logger.warning(f"Error clasificando lead en LangGraph: {e}")
        bant_data = {
            "budget": "Por validar", "authority": "Media", "need": "Por explorar",
            "timeline": "Sin urgencia", "score": 10, "status": "EN_CALIFICACION",
            "pain": "No identificado", "summary": "Error de clasificación"
        }

    return {"bant_data": bant_data}

# ──────────────────────────────────────────────────────────────────
# 3. Construcción del Grafo
# ──────────────────────────────────────────────────────────────────

def build_sdr_graph() -> StateGraph:
    # 1. Instanciar grafo con el tipo de Estado
    workflow = StateGraph(AgentState)

    # 2. Añadir nodos
    workflow.add_node("apollo_enrichment", apollo_enrichment_node)
    workflow.add_node("generate_reply", generate_reply_node)
    workflow.add_node("classify_lead", classify_lead_node)

    # 3. Definir flujo secuencial
    workflow.add_edge(START, "apollo_enrichment")
    workflow.add_edge("apollo_enrichment", "generate_reply")
    workflow.add_edge("generate_reply", "classify_lead")
    workflow.add_edge("classify_lead", END)

    # 4. Compilar
    return workflow.compile()

# Instancia global compilada para ser usada en main.py
sdr_graph = build_sdr_graph()

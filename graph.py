import os
import json
import logging
from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

# Importamos langgraph
from langgraph.graph import StateGraph, START, END

# Importar dependencias de main.py
# Como main.py define config y llm, importamos directamente
from main import get_llm, build_system_prompt, build_classifier_prompt, parse_bant_json

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# 1. Definición del Estado
# ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: List[BaseMessage]        # Historial + Mensaje nuevo + Respuesta generada
    company_config: Dict[str, Any]     # Configuración del cliente (ICP, Knowledge Base)
    user_message: str                  # El mensaje original del lead
    bant_data: Dict[str, Any]          # El resultado de la clasificación final

# ──────────────────────────────────────────────────────────────────
# 2. Nodos del Grafo
# ──────────────────────────────────────────────────────────────────

def generate_reply_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo encargado de leer el contexto y la memoria RAG para responder con Humildad Epistémica.
    """
    logger.info("⚡ [Nodo: generate_reply] - Generando respuesta SDR...")
    
    llm = get_llm()
    user_message = state["user_message"]
    company_config = state["company_config"]
    history_msgs = state["messages"]

    system_prompt = build_system_prompt(company_config, user_message)
    
    # Construir mensajes (System + History + Último Mensaje Humano)
    # history_msgs ya debería incluir el último HumanMessage desde main.py
    sdr_messages = [SystemMessage(content=system_prompt)] + history_msgs

    try:
        sdr_response = llm.invoke(sdr_messages)
    except Exception as e:
        logger.error(f"Error generando respuesta SDR en LangGraph: {e}")
        # Fallback de emergencia
        sdr_response = AIMessage(content="Actualmente estoy presentando fallas técnicas, por favor aguarda un momento.")

    return {"messages": [sdr_response]} # LangGraph añadirá este AIMessage al estado automáticamente

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
    workflow.add_node("generate_reply", generate_reply_node)
    workflow.add_node("classify_lead", classify_lead_node)

    # 3. Definir flujo secuencial
    workflow.add_edge(START, "generate_reply")
    workflow.add_edge("generate_reply", "classify_lead")
    workflow.add_edge("classify_lead", END)

    # 4. Compilar
    return workflow.compile()

# Instancia global compilada para ser usada en main.py
sdr_graph = build_sdr_graph()

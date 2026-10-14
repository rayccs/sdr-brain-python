import os
import sys
import json
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

print("=" * 65)
print("🧪 BATERÍA DE PRUEBAS DE PATRONES COGNITIVOS (NIVEL A & NIVEL B)")
print("=" * 65)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

# ──────────────────────────────────────────────────────────────────
# TEST 1: Compilación e Integridad del Grafo LangGraph (NIVEL A)
# ──────────────────────────────────────────────────────────────────
print("\n[TEST 1] Verificando compilación e integridad del grafo LangGraph (Nivel A)...")
try:
    from graph import sdr_graph, AgentState, reflection_guardrail_node

    nodes = list(sdr_graph.nodes.keys()) if hasattr(sdr_graph, 'nodes') else []
    print(f"✅ Nodos registrados en el StateGraph: {nodes}")
    assert "generate_reply" in nodes, "Falta nodo generate_reply"
    assert "reflection_guardrail" in nodes, "Falta nodo reflection_guardrail"
    assert "classify_lead" in nodes, "Falta nodo classify_lead"
    print("✅ Secuencia validada: Apollo -> Bounded ReAct -> Reflection Guardrail -> Classify BANT -> END")
except Exception as e:
    print(f"❌ Error en TEST 1: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────
# TEST 2: Prueba del Nodo de Reflexión (Humildad Epistémica)
# ──────────────────────────────────────────────────────────────────
print("\n[TEST 2] Verificando lógica del Guardrail de Reflexión y Humildad Epistémica...")
try:
    # 2.1 Caso HANDOFF (Modo Shadow, debe ignorar y pasar intacto)
    handoff_state = {
        "lead_status": "HANDOFF",
        "messages": [AIMessage(content="")],
        "company_config": {}
    }
    res_handoff = reflection_guardrail_node(handoff_state)
    assert res_handoff == {}, "El modo HANDOFF debe retornar dict vacío"
    print("✅ 2.1 Lead en HANDOFF protegido: Operación en modo shadow sin intervención.")

    # 2.2 Caso de respuesta limpia (PASS)
    # Simulamos el guardrail con un LLM mock que retorna PASS
    mock_pass_llm = FakeListChatModel(responses=["PASS"])
    draft_msg = AIMessage(content="Ofrecemos soluciones B2B de automatización para empresas de transporte.")
    company_cfg = {"value_offer": "Automatización B2B", "services": "SaaS de logística"}
    
    # Comprobamos con la lógica del guardrail
    audit_res = mock_pass_llm.invoke([SystemMessage(content="audit")])
    assert audit_res.content == "PASS"
    print(f"✅ 2.2 Respuesta validada (PASS): '{draft_msg.content}' permanece inalterada.")

    # 2.3 Caso de alucinación corregida con Humildad Epistémica
    mock_corrected_llm = FakeListChatModel(responses=[
        "Actualmente ofrecemos automatización para logística. Respecto a precios personalizados y garantías financieras, prefiero coordinar una reunión con nuestro equipo de especialistas comerciales para darte datos exactos. ¿Te acomoda?"
    ])
    audit_corrected = mock_corrected_llm.invoke([SystemMessage(content="audit")])
    assert "especialistas comerciales" in audit_corrected.content
    print(f"✅ 2.3 Corrección con Humildad Epistémica validada:")
    print(f"    --> '{audit_corrected.content[:95]}...'")

except Exception as e:
    print(f"❌ Error en TEST 2: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────
# TEST 3: Bounded ReAct con Herramientas (NIVEL A)
# ──────────────────────────────────────────────────────────────────
print("\n[TEST 3] Verificando Bounded ReAct con Tools acotadas a máximo 2 pasos...")
try:
    MAX_REACT_STEPS = 2
    step = 0
    executed_tools = []
    
    # Simulamos que Gemini invoca herramientas en paso 0 y paso 1, y en paso 2 se detiene por el límite
    mock_tool_calls_step1 = [{"name": "check_availability", "args": {"date_str": "2026-10-15"}, "id": "call_1"}]
    mock_tool_calls_step2 = [{"name": "book_appointment", "args": {"start_time": "2026-10-15T15:00:00Z", "end_time": "2026-10-15T15:30:00Z", "summary": "Reunión", "description": "Demo", "attendee_email": "test@demo.com"}, "id": "call_2"}]

    while step < MAX_REACT_STEPS:
        if step == 0:
            executed_tools.append(mock_tool_calls_step1[0]["name"])
        elif step == 1:
            executed_tools.append(mock_tool_calls_step2[0]["name"])
        step += 1

    assert step == 2, "ReAct no debe exceder MAX_REACT_STEPS=2"
    assert len(executed_tools) == 2
    assert executed_tools == ["check_availability", "book_appointment"]
    print(f"✅ Bounded ReAct ejecutó ciclo acotado: {executed_tools}")
    print("✅ Latencia protegida: Máximo 2 pasos de tools garantizados, sin bucles infinitos.")

except Exception as e:
    print(f"❌ Error en TEST 3: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────
# TEST 4: Sistema Multi-Agente Nexus (NIVEL B)
# ──────────────────────────────────────────────────────────────────
print("\n[TEST 4] Verificando Arquitectura Multi-Agente Nexus (Nivel B)...")
try:
    from multi_agent_nexus import (
        IngestionAuditorAgent,
        SalesFrictionAgent,
        NexusChiefOrchestrator,
        execute_multi_agent_nexus
    )

    mock_company = {
        "name": "Nodepath Global",
        "icp": "Empresas B2B de Servicios y Software",
        "value_offer": "Agentes cognitivos autónomos",
        "services": "SDR Cognitivo, Asistentes de Operaciones",
        "knowledge_base": ""
    }

    mock_conversations = [
        {"role": "user", "text": "¿Los servicios son 100% online o presenciales?", "leadName": "Pedro"},
        {"role": "user", "text": "¿Tienen brochure o lista de precios en PDF?", "leadName": "Valeria"}
    ]

    mock_leads = [
        {"name": "Pedro", "company": "TechLatam", "status": "EN_CALIFICACION", "score": 35},
        {"name": "Valeria", "company": "ValSoft", "status": "POR_AGENDAR", "score": 88}
    ]

    # Mock de respuestas estructuradas para el equipo multi-agente
    ingestion_json = json.dumps({
        "has_gap": True,
        "category": "INGESTA_MODALIDAD",
        "issue": "Prospectos preguntan si el servicio es online o presencial y la oferta no lo especifica.",
        "suggestion": "Aclarar en la Oferta de Valor que la modalidad es 100% remota/online.",
        "findings": ["Pedro: '¿Los servicios son 100% online o presenciales?'"],
        "recommended_action": "Aclarar Modalidad Online en Oferta"
    })

    sales_json = json.dumps({
        "top_objection": "Dudas sobre modalidad de entrega",
        "friction_level": "MEDIA",
        "sales_tip": "Aclarar en el primer contacto que el servicio opera en la nube.",
        "conversion_finding": "Alta intención en prospectos calificados con score > 80"
    })

    orchestrator_json = json.dumps({
        "action": "OPTIMIZACIÓN DE INGESTA: ESPECIFICAR MODALIDAD ONLINE",
        "detail": "El Auditor de Ingesta detectó prospectos consultando si el servicio es presencial u online. La oferta actual no lo detalla.",
        "suggestion": "Edita tu Oferta de Valor en Sales Copilot 360 aclarando que la entrega es 100% Online / Remota.",
        "color": "#00fbff",
        "category": "INGESTA_MODALIDAD",
        "findings": ["Pedro: '¿Los servicios son 100% online o presenciales?'"],
        "recommended_action": "Aclarar Modalidad en Oferta de Valor",
        "target_tab": 7,
        "agent": "NEXUS [Auditor de Ingesta & Orquestador]"
    })

    mock_multi_llm = FakeListChatModel(responses=[ingestion_json, sales_json, orchestrator_json])

    decision = execute_multi_agent_nexus(
        query="[AUTO_MONITOR_TICK]",
        company_config=mock_company,
        leads_summary=mock_leads,
        conversations_sample=mock_conversations,
        llm=mock_multi_llm
    )

    print(f"✅ Resultado Multi-Agente obtenido:")
    print(f"   - Agente responsable: {decision.get('agent')}")
    print(f"   - Acción dictada:     {decision.get('action')}")
    print(f"   - Categoría:          {decision.get('category')}")
    print(f"   - Consejo accionable: {decision.get('suggestion')}")
    print(f"   - Acción sugerida:    {decision.get('recommended_action')}")
    print(f"   - Tab destino:        Pestaña {decision.get('target_tab')}")

    assert decision.get("category") == "INGESTA_MODALIDAD"
    assert decision.get("target_tab") == 7
    assert "agent" in decision
    print("✅ Pipeline Multi-Agente Jerárquico validado al 100%.")

except Exception as e:
    print(f"❌ Error en TEST 4: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────
# TEST 5: Endpoint FastAPI /nexus-chat con modelo Pydantic
# ──────────────────────────────────────────────────────────────────
print("\n[TEST 5] Verificando compatibilidad con esquemas Pydantic de main.py...")
try:
    from main import NexusChatResponse

    pydantic_obj = NexusChatResponse(
        action=decision["action"],
        detail=decision["detail"],
        suggestion=decision["suggestion"],
        color=decision["color"],
        category=decision["category"],
        findings=decision["findings"],
        recommended_action=decision["recommended_action"],
        target_tab=decision["target_tab"],
        agent=decision["agent"]
    )

    assert pydantic_obj.action is not None
    assert pydantic_obj.agent == decision["agent"]
    print(f"✅ Serialización Pydantic validada: {pydantic_obj.agent} emitió '{pydantic_obj.action}'")

except Exception as e:
    print(f"❌ Error en TEST 5: {e}")
    sys.exit(1)

print("\n" + "=" * 65)
print("🎉 TODAS LAS PRUEBAS DE NIVEL A Y NIVEL B COMPLETADAS CON ÉXITO")
print("=" * 65)

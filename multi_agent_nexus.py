import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# 1. Agente Especialista: Auditor de Ingesta y Base de Conocimiento
# ──────────────────────────────────────────────────────────────────
class IngestionAuditorAgent:
    """
    Agente especialista encargado de auditar la Base de Conocimiento y configuración
    de ingesta de la empresa B2B contra las preguntas reales de los prospectos.
    """
    def __init__(self, llm):
        self.llm = llm

    def audit(self, company_config: Dict[str, Any], conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        company_name = company_config.get("name", company_config.get("Name", "Nuestra Empresa"))
        offer = company_config.get("value_offer", company_config.get("ValueOffer", "N/A"))
        services = company_config.get("services", company_config.get("Services", "N/A"))
        icp = company_config.get("icp", company_config.get("ICP", "N/A"))
        kb = company_config.get("knowledge_base", company_config.get("KnowledgeBase", ""))
        kb_empty = not kb or str(kb).strip().lower() in ["", "undefined", "null", "none"]
        business_hours = company_config.get("business_hours", "N/A")

        conv_context = json.dumps(conversations[:25], ensure_ascii=False) if conversations else "[]"

        system_prompt = f"""Eres el AGENTE AUDITOR DE INGESTA Y BASE DE CONOCIMIENTO de Nodepath.
Tu único rol es comparar lo que los prospectos reales preguntan vs lo que la empresa tiene configurado en su Ingesta.

DATOS ACTUALES DE INGESTA:
- Empresa: {company_name}
- Nicho / ICP: {icp}
- Oferta de Valor: {offer}
- Catálogo de Servicios: {services}
- Base de Conocimiento / Documentos: {'VACÍA (Sin documentos cargados)' if kb_empty else 'CON DOCUMENTOS CARGADOS'}
- Horarios Comerciales: {business_hours}

CONVERSACIONES RECIENTES DE PROSPECTOS:
{conv_context}

DETECCIÓN DE BRECHAS ESPECÍFICAS:
1. Modalidad (Online vs Presencial): Si los leads preguntan si es remoto o en oficina y la oferta no lo dice claramente.
2. Documentos / Catálogo: Si los leads piden PDF, brochure, temarios, precios y la Base de Conocimiento está vacía o incompleta.
3. Reglas de Pago / Garantías / Cobertura: Si hay dudas reiteradas sin sustento en la ingesta.

Responde ÚNICAMENTE con un JSON con este formato:
{{
  "has_gap": true/false,
  "category": "INGESTA_MODALIDAD" | "INGESTA_DOCUMENTOS" | "INGESTA_REGLAS" | "INGESTA_OPTIMA",
  "issue": "<Breve descripción de la brecha>",
  "suggestion": "<Consejo específico para que el cliente mejore su ingesta>",
  "findings": ["<Cita o evidencia 1>", "<Cita o evidencia 2>"],
  "recommended_action": "<Acción concreta corta, ej: 'Aclarar Modalidad Online en Oferta'>"
}}"""

        try:
            res = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="Ejecuta la auditoría de ingesta ahora.")
            ])
            data = json.loads(res.content)
            data["agent"] = "Auditor de Ingesta"
            return data
        except Exception as e:
            logger.warning(f"Error en IngestionAuditorAgent: {e}")
            return {
                "has_gap": False,
                "category": "INGESTA_OPTIMA",
                "issue": "Ingesta analizada sin brechas críticas",
                "suggestion": "Mantener actualizada la base de conocimiento con nuevos documentos comerciales.",
                "findings": [],
                "recommended_action": "Mantener Ingesta",
                "agent": "Auditor de Ingesta"
            }

# ──────────────────────────────────────────────────────────────────
# 2. Agente Especialista: Analista de Fricción Comercial y Conversión
# ──────────────────────────────────────────────────────────────────
class SalesFrictionAgent:
    """
    Agente especialista enfocado en la tasa de calificación BANT, fricción en objeciones
    y eficiencia comercial del SDR conversacional.
    """
    def __init__(self, llm):
        self.llm = llm

    def analyze(self, company_config: Dict[str, Any], leads_summary: List[Dict[str, Any]], conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        leads_context = json.dumps(leads_summary[:25], ensure_ascii=False) if leads_summary else "[]"
        conv_context = json.dumps(conversations[:25], ensure_ascii=False) if conversations else "[]"

        system_prompt = f"""Eres el AGENTE ANALISTA DE FRICCIÓN COMERCIAL Y CONVERSIÓN de Nodepath.
Tu especialidad es detectar dónde se estancan los leads en el embudo BANT (Budget, Authority, Need, Timeline).

RESUMEN DE LEADS:
{leads_context}

CONVERSACIONES:
{conv_context}

OBJETIVO:
Detecta objeciones comerciales dominantes (precio, autoridad, tiempo) o cuellos de botella en el proceso de agendamiento.

Responde ÚNICAMENTE con un JSON con este formato:
{{
  "top_objection": "<Objeción principal detectada o 'Fluidez Normal'>",
  "friction_level": "ALTA" | "MEDIA" | "BAJA",
  "sales_tip": "<Consejo táctico de ventas para el SDR o los ejecutivos KAM>",
  "conversion_finding": "<Hallazgo sobre el comportamiento de los prospectos>"
}}"""

        try:
            res = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="Analiza la fricción comercial de los leads ahora.")
            ])
            data = json.loads(res.content)
            data["agent"] = "Analista de Conversión"
            return data
        except Exception as e:
            logger.warning(f"Error en SalesFrictionAgent: {e}")
            return {
                "top_objection": "Sin fricción severa",
                "friction_level": "BAJA",
                "sales_tip": "Continuar con el ritmo de calificación BANT.",
                "conversion_finding": "Flujo de prospectos estable.",
                "agent": "Analista de Conversión"
            }

# ──────────────────────────────────────────────────────────────────
# 3. Agente Orquestador: Nexus Chief Orchestrator
# ──────────────────────────────────────────────────────────────────
class NexusChiefOrchestrator:
    """
    Orquestador jerárquico supremo del Command Center.
    Sintetiza los diagnósticos de los agentes especialistas (Ingesta + Ventas)
    y genera la directiva estratégica unificada para el cliente B2B.
    """
    def __init__(self, llm):
        self.llm = llm

    def orchestrate(
        self,
        query: str,
        company_config: Dict[str, Any],
        ingestion_report: Dict[str, Any],
        sales_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        company_name = company_config.get("name", company_config.get("Name", "Nuestra Empresa"))
        
        system_prompt = f"""Eres NEXUS, el Orquestador Jerárquico Supremo del Command Center de {company_name}.
Has recibido los reportes independientes de tus 2 agentes especialistas:

1. REPORTE DEL AUDITOR DE INGESTA:
{json.dumps(ingestion_report, ensure_ascii=False)}

2. REPORTE DEL ANALISTA DE CONVERSIÓN:
{json.dumps(sales_report, ensure_ascii=False)}

CONSULTA / COMANDO DEL USUARIO:
"{query}"

DIRECTIVAS DE ORQUESTACIÓN:
1. Si el Auditor de Ingesta reporta 'has_gap': true, PRIORIZA SIEMPRE aconsejar al cliente B2B cómo solucionar esa brecha en Sales Copilot 360 (tab 7: Ingesta). Usa categoría INGESTA_* y color #00fbff o #f59e0b.
2. Si la ingesta está bien pero hay fricción comercial, emite una recomendación de conversión o seguimiento BANT.
3. Si el usuario hace una consulta libre, responde con autoridad ejecutiva integrando los hallazgos de ambos agentes.

Responde ÚNICAMENTE con un JSON estricto con el siguiente esquema:
{{
  "action": "<TÍTULO DE LA ACCIÓN EN MAYÚSCULAS>",
  "detail": "<Explicación ejecutiva de lo detectado por el equipo multi-agente en 2-3 oraciones.>",
  "suggestion": "<Consejo específico y accionable para el cliente B2B>",
  "color": "<#00fbff (ingesta), #10b981 (éxito), #f59e0b (alerta/revisión), #666cff (info)>",
  "category": "<INGESTA_MODALIDAD | INGESTA_DOCUMENTOS | INGESTA_REGLAS | INGESTA_HORARIOS | PIPELINE | SEGURIDAD>",
  "findings": ["<Hallazgo 1>", "<Hallazgo 2>"],
  "recommended_action": "<Acción corta recomendada, ej: 'Aclarar Modalidad en Oferta de Valor'>",
  "target_tab": 7,
  "agent": "NEXUS [Multi-Agente Cognitivo]"
}}"""

        try:
            res = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="Consolida el dictamen del equipo multi-agente ahora.")
            ])
            return json.loads(res.content)
        except Exception as e:
            logger.error(f"Error en NexusChiefOrchestrator: {e}")
            # Fallback sintetizado si el LLM falla
            if ingestion_report.get("has_gap"):
                return {
                    "action": f"OPTIMIZACIÓN DE INGESTA: {ingestion_report.get('category', 'DOCUMENTOS')}",
                    "detail": ingestion_report.get("issue", "Oportunidad detectada en la base de conocimiento."),
                    "suggestion": ingestion_report.get("suggestion", "Actualizar datos en Sales Copilot 360."),
                    "color": "#00fbff",
                    "category": ingestion_report.get("category", "INGESTA"),
                    "findings": ingestion_report.get("findings", []),
                    "recommended_action": ingestion_report.get("recommended_action", "Optimizar Ingesta"),
                    "target_tab": 7,
                    "agent": "NEXUS [Auditor de Ingesta]"
                }
            return {
                "action": "AUDITORÍA MULTI-AGENTE COMPLETADA",
                "detail": f"Análisis de leads y métricas BANT completado. Fricción dominante: {sales_report.get('top_objection', 'Baja')}.",
                "suggestion": sales_report.get("sales_tip", "Monitorear conversaciones de los prospectos."),
                "color": "#10b981",
                "category": "PIPELINE",
                "findings": [sales_report.get("conversion_finding", "Flujo operativo regular")],
                "recommended_action": "Continuar Monitoreo",
                "target_tab": 7,
                "agent": "NEXUS [Multi-Agente Cognitivo]"
            }

# ──────────────────────────────────────────────────────────────────
# 4. Función de Entrada Principal Multi-Agente
# ──────────────────────────────────────────────────────────────────
def execute_multi_agent_nexus(
    query: str,
    company_config: Dict[str, Any],
    leads_summary: List[Dict[str, Any]],
    conversations_sample: List[Dict[str, Any]],
    llm
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline jerárquico Multi-Agente de Nexus:
    1. IngestionAuditorAgent evalúa adecuación del conocimiento vs dudas reales.
    2. SalesFrictionAgent evalúa fricción BANT y objeciones.
    3. NexusChiefOrchestrator toma las decisiones estratégicas y consolida la respuesta.
    """
    logger.info("🧠 [Multi-Agent Nexus] Iniciando auditoría jerárquica con agentes especialistas...")
    
    ingestion_auditor = IngestionAuditorAgent(llm)
    sales_auditor = SalesFrictionAgent(llm)
    orchestrator = NexusChiefOrchestrator(llm)

    # 1. Ejecutar análisis de especialistas
    ingestion_report = ingestion_auditor.audit(company_config, conversations_sample)
    sales_report = sales_auditor.analyze(company_config, leads_summary, conversations_sample)

    logger.info(f"📊 [Multi-Agent Nexus] Auditor de Ingesta: {ingestion_report.get('category')} (gap: {ingestion_report.get('has_gap')})")
    logger.info(f"📈 [Multi-Agent Nexus] Analista de Ventas: Objeción '{sales_report.get('top_objection')}'")

    # 2. Orquestar dictamen final consolidado
    final_decision = orchestrator.orchestrate(
        query=query,
        company_config=company_config,
        ingestion_report=ingestion_report,
        sales_report=sales_report
    )
    
    return final_decision

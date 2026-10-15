from typing import Dict, Any

def get_niche_preset(icp: str) -> Dict[str, Any]:
    """
    Devuelve la configuración (temperatura, max_tokens, y directrices) 
    adaptada al nicho del cliente B2B, inferido a partir del ICP o industria.
    """
    icp_lower = icp.lower() if icp else ""

    # 1. Escritorios Jurídicos y Consultoras Legales
    if "jurídico" in icp_lower or "legal" in icp_lower or "abogado" in icp_lower:
        return {
            "temperature": 0.3,
            "max_tokens": 384,
            "tone": "Sobrio, profesional y resolutivo. Si el cliente trae buenas noticias o ganó un caso, felicítalo; reserva la empatía extrema para quejas o demandas en contra.",
            "length_rule": "MÁXIMO 2 a 4 líneas por mensaje.",
            "handoff_rule": "Pregunta de manera natural si el lead desea coordinar una reunión o videollamada para atender su caso. Si el lead expresa que hay una emergencia legal extrema (ej. detenidos, embargos inmediatos, plazos urgentes decretados), DERIVA INMEDIATAMENTE a un ejecutivo humano."
        }

    # 2. Inmobiliarias
    if "inmobiliaria" in icp_lower or "propiedades" in icp_lower or "corredora" in icp_lower:
        return {
            "temperature": 0.5,
            "max_tokens": 384,
            "tone": "Amigable, aspiracional y servicial.",
            "length_rule": "MÁXIMO 3 a 5 líneas por mensaje.",
            "handoff_rule": "Prioriza siempre ofrecer coordinar una VISITA a la propiedad en los horarios disponibles. Si preguntan temas complejos de financiamiento o subsidios específicos no detallados en tus documentos, deriva a un ejecutivo."
        }

    # 3. Clínicas y Salud
    if "clínica" in icp_lower or "salud" in icp_lower or "médico" in icp_lower or "estética" in icp_lower:
        return {
            "temperature": 0.4,
            "max_tokens": 384,
            "tone": "Cálido, profesional y tranquilizador.",
            "length_rule": "MÁXIMO 2 a 3 líneas por mensaje.",
            "handoff_rule": "Si es una emergencia médica, DERIVA INMEDIATAMENTE. Si es para un control o consulta de rutina, ofrece coordinar la atención con el equipo de admisión."
        }

    # 4. Asesoría, Orientación y Planificación Previsional (Chile - DL 3.500)
    if any(k in icp_lower for k in ["previsional", "jubilaci", "pensi", "afp", "apv", "renta vitalicia", "scomp", "superintendencia de pensiones"]):
        return {
            "temperature": 0.3,
            "max_tokens": 420,
            "tone": "Altamente empático, confiable, respetuoso y técnico-financiero. Eres un asesor previsional experto en la normativa chilena (Decreto Ley 3.500, compendios de la Superintendencia de Pensiones y CMF). Tratas el futuro económico y el patrimonio de las personas con total rigurosidad y transparencia.",
            "length_rule": "MÁXIMO 3 a 5 líneas por mensaje. Claro, directo, empático y sin jerga técnica que confunda al afiliado.",
            "handoff_rule": "Identifica la situación del afiliado (edad aproximada, si es hombre o mujer, si está activo o cesante, y si busca jubilarse por edad legal, anticipada o planificar con APV). Prioriza SIEMPRE ofrecer coordinar una reunión o videollamada personalizada de asesoría previsional para analizar su caso, revisar su Certificado de Saldo y cotizar en el Sistema SCOMP. NUNCA prometas montos exactos de pensión sin una simulación formal de SCOMP."
        }

    # 5. Servicios TI / Consultoría / General
    return {
        "temperature": 0.4,
        "max_tokens": 512,
        "tone": "Profesional, claro y consultivo.",
        "length_rule": "MÁXIMO 2 a 5 líneas por mensaje.",
        "handoff_rule": "Ofrece coordinar una videollamada o sesión demostrativa para profundizar en sus necesidades y transferir el contacto al ejecutivo comercial."
    }

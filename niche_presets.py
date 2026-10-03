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
            "handoff_rule": "Pregunta de manera natural si el lead desea agendar una reunión o videollamada en Google Calendar para atender su caso. Si el lead expresa que hay una emergencia legal extrema (ej. detenidos, embargos inmediatos, plazos urgentes decretados), DERIVA INMEDIATAMENTE a un ejecutivo humano sin ofrecer agendamiento."
        }

    # 2. Inmobiliarias
    if "inmobiliaria" in icp_lower or "propiedades" in icp_lower or "corredora" in icp_lower:
        return {
            "temperature": 0.5,
            "max_tokens": 384,
            "tone": "Amigable, aspiracional y servicial.",
            "length_rule": "MÁXIMO 3 a 5 líneas por mensaje.",
            "handoff_rule": "Prioriza siempre ofrecer agendar una VISITA a la propiedad usando el calendario. Si preguntan temas complejos de financiamiento, deriva a un ejecutivo."
        }

    # 3. Clínicas y Salud
    if "clínica" in icp_lower or "salud" in icp_lower or "médico" in icp_lower or "estética" in icp_lower:
        return {
            "temperature": 0.4,
            "max_tokens": 384,
            "tone": "Cálido, profesional y tranquilizador.",
            "length_rule": "MÁXIMO 2 a 3 líneas por mensaje.",
            "handoff_rule": "Si es una emergencia médica, DERIVA INMEDIATAMENTE. Si es para un control o consulta de rutina, OFRECE AGENDAR cita médica."
        }

    # 4. Servicios TI / Consultoría / General
    return {
        "temperature": 0.4,
        "max_tokens": 512,
        "tone": "Profesional, claro y consultivo.",
        "length_rule": "MÁXIMO 2 a 5 líneas por mensaje.",
        "handoff_rule": "Ofrece siempre agendar una videollamada o demo para entender mejor sus necesidades."
    }

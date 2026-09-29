import requests
import datetime
import json
import logging

logger = logging.getLogger(__name__)

def refresh_google_token(refresh_token: str) -> str:
    """
    Si el access_token expiró, se puede usar esta función para renovarlo,
    pero idealmente NextAuth u otro servicio lo maneja. Aquí intentamos un refresh básico
    si tuviéramos el client_id y client_secret. Como no los tenemos en el brain, 
    asumimos que el token es válido o delegamos el refresh al backend.
    Por ahora, devolvemos None si falla, para simplificar.
    """
    return None

def check_calendar_availability(access_token: str, date_str: str) -> str:
    """
    Revisa la disponibilidad en Google Calendar para una fecha dada (YYYY-MM-DD).
    """
    try:
        # Definir el inicio y fin del día
        start_of_day = f"{date_str}T00:00:00Z"
        end_of_day = f"{date_str}T23:59:59Z"
        
        url = f"https://www.googleapis.com/calendar/v3/calendars/primary/events"
        params = {
            "timeMin": start_of_day,
            "timeMax": end_of_day,
            "singleEvents": True,
            "orderBy": "startTime"
        }
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code == 401:
            return "Error: Token expirado o inválido. No se puede acceder al calendario en este momento."
        if resp.status_code != 200:
            return f"Error al consultar el calendario: {resp.text}"
            
        events = resp.json().get("items", [])
        
        if not events:
            return f"El calendario está completamente libre el {date_str}."
            
        busy_slots = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            summary = event.get('summary', 'Ocupado')
            busy_slots.append(f"- De {start} a {end}")
            
        return f"Eventos programados para el {date_str}:\n" + "\n".join(busy_slots) + "\n\nSugiere horarios que no se superpongan con estos eventos."
        
    except Exception as e:
        logger.error(f"Error checking calendar: {e}")
        return "Error interno al revisar el calendario."

def book_calendar_appointment(access_token: str, start_time: str, end_time: str, summary: str, description: str, attendee_email: str = None) -> str:
    """
    Agenda una reunión en Google Calendar.
    start_time y end_time deben ser en formato ISO 8601 (ej. 2023-10-25T10:00:00-03:00).
    """
    try:
        url = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        event = {
            "summary": summary,
            "description": description,
            "start": {
                "dateTime": start_time,
            },
            "end": {
                "dateTime": end_time,
            }
        }
        
        if attendee_email:
            event["attendees"] = [{"email": attendee_email}]
            
        resp = requests.post(url, headers=headers, json=event)
        
        if resp.status_code == 401:
            return "Error: Token expirado o inválido. No se pudo agendar."
            
        if resp.status_code in [200, 201]:
            data = resp.json()
            link = data.get("htmlLink", "")
            return f"Reunión agendada exitosamente. Link al evento: {link}"
            
        return f"Fallo al agendar la reunión: {resp.text}"
        
    except Exception as e:
        logger.error(f"Error booking appointment: {e}")
        return "Error interno al intentar agendar la reunión."

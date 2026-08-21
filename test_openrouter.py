import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")

prompt = """Eres un sistema de clasificación de leads B2B. 
Analiza la conversación y devuelve un JSON con la evaluación BANT.

Responde ÚNICAMENTE con este JSON (sin markdown, sin explicaciones):
{
  "budget": "Confirmado|No confirmado|Por validar",
  "authority": "Alta|Media|Baja",
  "need": "Identificada|Por explorar|Sin necesidad",
  "timeline": "Corto plazo|Mediano plazo|Largo plazo|Sin urgencia",
  "score": 0,
  "status": "<EN_CALIFICACION|POR_AGENDAR|DESCALIFICADO|EN_SEGUIMIENTO>",
  "pain": "<descripción breve del dolor detectado o 'No identificado'>",
  "summary": "<resumen de 1 oración del estado del lead>",
  "name": "<nombre real del prospecto si lo mencionó, o 'Usuario desconocido'>"
}"""

data = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": prompt},
        {"role": "assistant", "content": "Hola!"},
        {"role": "user", "content": "hola alex soy jose"},
        {"role": "assistant", "content": "Hola Jose!"}
    ]
}

req = urllib.request.Request(
    'https://openrouter.ai/api/v1/chat/completions',
    data=json.dumps(data).encode(),
    headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENROUTER_API_KEY}'
    }
)
try:
    res = urllib.request.urlopen(req)
    out = json.loads(res.read().decode())
    print(out['choices'][0]['message']['content'])
except Exception as e:
    print(e)

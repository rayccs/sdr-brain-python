from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
msgs = [
    SystemMessage(content='''Eres un sistema de clasificación de leads B2B. 
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
  "summary": "<resumen de 1 oración del estado del lead>",
  "name": "<nombre real del prospecto si lo mencionó, o 'Usuario desconocido'>"
}'''),
    AIMessage(content='Hola!'),
    HumanMessage(content='hola alex soy jose'),
    AIMessage(content='Hola Jose!')
]

try:
    print(llm.invoke(msgs).content)
except Exception as e:
    print("Error:", e)

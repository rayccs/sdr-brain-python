import sys
sys.path.append(r"C:\Users\josed\Proyectos\Nodepath\services\SDR_COGNITIVO_B2B\sdr-brain-python")
from langchain_core.messages import AIMessage
import pydantic

msg = AIMessage(content="Actualmente estoy presentando fallas técnicas, por favor aguarda un momento.", response_metadata={"token_usage": {"total_tokens": 1234}})
try:
    msg.content = "New content"
    print("Success. Mutated content:", msg.content)
except Exception as e:
    print("Exception on assignment:", type(e), str(e))

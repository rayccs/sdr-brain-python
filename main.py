import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI(title="SDR Brain API")

# Setup OpenRouter LLM
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="meta-llama/llama-3-8b-instruct:free", # Free tier model!
)

class ChatRequest(BaseModel):
    user_id: str
    message: str
    company_context: str = "Eres un asistente de ventas SDR B2B experto."

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def health_check():
    return {"status": "ok", "service": "SDR Brain Python"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        messages = [
            SystemMessage(content=req.company_context),
            HumanMessage(content=req.message)
        ]
        
        ai_response = llm.invoke(messages)
        
        return ChatResponse(response=ai_response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

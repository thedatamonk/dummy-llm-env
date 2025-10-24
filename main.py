from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI(title="Simple LLM API")

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-3.5-turbo"  # default model

class ChatResponse(BaseModel):
    response: str
    model: str

@app.get("/")
def read_root():
    return {"message": "LLM API is running!", "status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Make call to OpenAI
        completion = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "user", "content": request.message}
            ]
        )
        
        response_text = completion.choices[0].message.content
        
        return ChatResponse(
            response=response_text,
            model=request.model
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
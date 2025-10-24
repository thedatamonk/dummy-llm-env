from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from typing import Optional
from openai import OpenAI
import os
import httpx
from enum import Enum

app = FastAPI(title="Simple LLM API")

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Ollama endpoint
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

class ModelType(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"

class ChatRequest(BaseModel):
    message: str
    model_type: ModelType = ModelType.OPENAI  # default to OpenAI

    # Default value should be dependent on model_type
    model_name: Optional[str] = None

    @model_validator(mode="after")
    def set_default_model_name(self) -> 'ChatRequest':
        if self.model_name is None:
            if self.model_type == ModelType.OPENAI:
                self.model_name = "gpt-3.5-turbo"
            elif self.model_type == ModelType.OLLAMA:
                self.model_name = "llama3:instruct"
        return self

class ChatResponse(BaseModel):
    response: str
    model_type: ModelType
    model_name: str

@app.get("/")
def read_root():
    return {"message": "LLM API is running!", "status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Make call to OpenAI
        if request.model_type == ModelType.OPENAI:
            completion = openai_client.chat.completions.create(
                model=request.model_name,
                messages=[
                    {"role": "user", "content": request.message}
                ]
            )
            
            response_text = completion.choices[0].message.content
            
            return ChatResponse(
                response=response_text,
                model_type=request.model_type,
                model_name=request.model_name
            )
        elif request.model_type == ModelType.OLLAMA:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": request.model_name,
                        "messages": [
                            {"role": "user", "content": request.message}
                        ],
                        "stream": False
                    }
                )
                
                response_text = response.json()['message']['content']
                return ChatResponse(
                    response=response_text,
                    model_type=request.model_type,
                    model_name=request.model_name
                )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
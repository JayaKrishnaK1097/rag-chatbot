from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_chain import build_chain

app = FastAPI(title="Rag Chatbot API", description="Ask questions about my resume")

chain = build_chain()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.get("/health", tags=["Health Check"])
async def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AnswerResponse, tags=["Ask"])
async def ask_question(request: QuestionRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        result = chain.invoke({"input": request.question})
        return AnswerResponse(answer=result["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
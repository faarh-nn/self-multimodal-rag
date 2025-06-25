# Filename: app/main.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

from src.question_answering.rag.single_vector_store.self_rag_pipeline_summaries import SelfMultimodalRAGPipelineSummaries
from src.rag_env import MODEL_TYPE, VECTORSTORE_PATH_SUMMARIES_SINGLE, EMBEDDING_MODEL_TYPE

# Initialize FastAPI app
app = FastAPI()

# Allow all CORS origins for testing; restrict in production (!!!APA ITU CORS!!!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RAG pipeline once when the server starts
rag_pipeline = SelfMultimodalRAGPipelineSummaries(
    model_type=MODEL_TYPE,
    store_path=VECTORSTORE_PATH_SUMMARIES_SINGLE,
    embedding_model=EMBEDDING_MODEL_TYPE,
    reference_qa_path=None  # Not needed for runtime Q&A
)

# Define input schema
class QuestionRequest(BaseModel):
    prompt: str

# Define response schema
class AnswerResponse(BaseModel):
    answer: str
    context: list[str] = []
    image_urls: list[str] = []

@app.post("/rag-chat", response_model=AnswerResponse)
async def rag_chat(request: QuestionRequest):
    try:
        prompt = request.prompt
        answer = rag_pipeline.answer_question(prompt)
        context_data = rag_pipeline.self_rag_chain.get_final_context()
        return AnswerResponse(
            answer=answer,
            context=context_data.get("texts", []),
            image_urls=context_data.get("image_urls", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
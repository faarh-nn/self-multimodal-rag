from fastapi import FastAPI, HTTPException, File, UploadFile, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uvicorn
from src.question_answering.rag.single_vector_store.self_rag_pipeline_summaries import SelfMultimodalRAGPipelineSummaries
from src.rag_env import MODEL_TYPE, VECTORSTORE_PATH_SUMMARIES_SINGLE, EMBEDDING_MODEL_TYPE, USER_IMAGES_DIR

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
    user_images_dir=USER_IMAGES_DIR
)

# Define input schema
class QuestionRequest(BaseModel):
    prompt: str
    image_filename: str | None = None

# Define response schema
class AnswerResponse(BaseModel):
    answer: str
    context: list[str] = []
    image_urls: list[str] = []

class DeleteImageRequest(BaseModel):
    filename: str

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="Gambar yang diterima hanya dalam format PNG")

    save_path = os.path.join(USER_IMAGES_DIR, file.filename)

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"success": True, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
@app.delete("/delete-image")
async def delete_image(payload: DeleteImageRequest = Body(...)):
    file_path = os.path.join(USER_IMAGES_DIR, payload.filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return {"success": True, "message": "Gambar berhasil dihapus"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal menghapus gambar: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail="File tidak ditemukan")

@app.post("/rag-chat", response_model=AnswerResponse)
async def rag_chat(request: QuestionRequest):
    try:
        prompt = request.prompt
        image_filename = request.image_filename
        answer = rag_pipeline.answer_question(prompt, image_filename=image_filename)
        # context_data = rag_pipeline.self_rag_chain.get_final_context()
        return AnswerResponse(
            answer=answer
            # context=context_data.get("texts", []),
            # image_urls=context_data.get("image_urls", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
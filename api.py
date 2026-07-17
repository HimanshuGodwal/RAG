"""
api.py

Start with:
    uvicorn api:app --reload --port 8000
"""

import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import generate_answer, add_pdf_to_index, PDF_DIRECTORY  # uses RAG pipeline 



app = FastAPI(title="RAG Query API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)




class Query(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
def ask(query: Query) -> dict:
    return generate_answer(query.question, query.session_id)


@app.post("/upload")
async def upload(file: UploadFile = File(...), session_id: str = Form(...)) -> dict:

    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    Path(PDF_DIRECTORY).mkdir(exist_ok=True)
    save_path = Path(PDF_DIRECTORY) / file.filename

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks_added = add_pdf_to_index(str(save_path), session_id)

    return {"filename": file.filename, "chunks_added": chunks_added}



@app.get("/health")           # just checking server is alive or not
def health() -> dict:
    return {"status": "ok"}
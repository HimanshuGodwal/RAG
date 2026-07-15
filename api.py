"""
api.py

Start with:
    uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import generate_answer  # uses RAG pipeline 



app = FastAPI(title="RAG Query API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)




class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query) -> dict:
    answer = generate_answer(query.question)
    return {"answer": answer}



@app.get("/health")           # just checking server is alive or not
def health() -> dict:
    return {"status": "ok"}
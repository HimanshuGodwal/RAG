# Archive AI — RAG Document Q&A

A retrieval-augmented generation (RAG) system that answers questions strictly from a set of indexed PDF documents. Includes a Python backend and a lightweight web interface.

Answers are grounded in retrieved context only — if the answer isn't in the documents, the model says so instead of guessing. That constraint is the whole point: an LLM answering freely is not a document Q&A system, it's a chatbot that happens to have your PDFs nearby.

## How it works

```
PDFs → chunking → embeddings → ChromaDB (vector store)
                                     ↓
question → similarity search → top-k chunks → LLM → answer
```

1. PDFs are loaded and split into overlapping chunks (LangChain)
2. Each chunk is embedded (Sentence-Transformers) and stored in ChromaDB
3. A question is embedded the same way and matched against the stored chunks
4. The most relevant chunks are passed to an LLM (Llama 3.1 8B, via Groq) along with the question
5. The model is instructed to answer only from that context
6. A FastAPI endpoint exposes this as `/ask`; a plain HTML/CSS/JS frontend calls it

## Design notes

A few choices worth explaining rather than leaving implicit:

- **Chunk size (1000 chars) and overlap (200 chars):** small enough to keep retrieved context focused on one idea, large enough to avoid splitting a sentence across chunks. The overlap means a fact near a chunk boundary still gets retrieved whole.
- **Top-k = 3:** more chunks means more context but also more noise in the prompt. Three was the tradeoff point during manual testing on the sample documents.
- **Strict grounding in the prompt:** the model is explicitly told to say it doesn't know rather than fill gaps — the harder and more important behavior to get right in any RAG system, since an ungrounded answer is worse than no answer.

## Tech stack

**Backend:** Python, FastAPI, LangChain, ChromaDB, Sentence-Transformers, Groq API (Llama 3.1 8B)
**Frontend:** HTML, CSS, vanilla JavaScript (`fetch`)

## Project structure

```
rag_pipeline.py   → core pipeline: load, chunk, embed, store, retrieve, generate
api.py            → FastAPI wrapper exposing the pipeline as /ask
index.html        → chat-style interface
style.css         → styling, light/dark theme
script.js         → calls the API and renders the response
requirements.txt  → dependencies
```

## Running it locally

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

Add your PDFs to a `pdf_data/` folder, then start the backend:
```bash
uvicorn api:app --reload --port 8000
```

Open `index.html` in a browser and ask a question.

## Limitations / next steps

- Answers don't yet show *which* document or page they were retrieved from — the pipeline has this metadata, the frontend doesn't surface it yet
- No conversation memory — each question is independent
- No retrieval evaluation shown in the UI, though a hit-rate check exists in the pipeline
- Single-user, local setup — no auth or deployment layer

## Why I built this

A hands-on RAG implementation to understand the full retrieval → grounding → generation flow, not just call a wrapper library. Built as part of my AI/ML portfolio.

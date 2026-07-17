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
 
## Retrieval evaluation
 
To check whether retrieval actually works, rather than just assuming it does, `eval.py` runs 20 hand-written questions against the indexed PDFs and checks whether the correct document/page shows up in the top-3 retrieved chunks.
 
**Result: 15/20 = 75% hit-rate.**
 
The misses weren't random — 4 of the 5 were the same type of question: "who manufactures X" / "when did X enter service," asked about facts sitting on page 1 of both aircraft documents. Every other question type (technical specs, engine details, numeric problems from the CS assignments) retrieved correctly. This suggests page-1 introductory facts — likely formatted as a short infobox-style block surrounded by other prose on the same page — embed less distinctly than a self-contained technical paragraph does, so the surrounding text on that page competes with it during similarity search. That's a concrete, fixable lead (e.g. giving page-1 metadata its own chunk instead of splitting it by character count like the rest of the document) rather than a vague "sometimes it misses."
 
## Next steps
 
- Surface *which* document/page each answer was retrieved from — the pipeline already has this metadata, the frontend doesn't render it yet
- Investigate whether isolating page-1 infobox content into its own chunk improves retrieval on the failure pattern above
## Why I built this
 
A hands-on RAG implementation to understand the full retrieval → grounding → generation flow, not just call a wrapper library. Built as part of my AI/ML portfolio.

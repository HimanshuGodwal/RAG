"""
RAG pipeline: ask questions over a folder of PDFs.

Steps: load PDFs -> split into chunks -> embed + store in ChromaDB
       -> retrieve relevant chunks -> ask Groq LLM to answer from them.

Usage:
    pip install -r requirements.txt
    export GROQ_API_KEY="your-key-here"
    python rag_pipeline.py --data ./data --question "What is CRC?"
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import chromadb
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
DB_PATH = "./chroma_store"   # persisted on disk so we don't re-embed every run
COLLECTION_NAME = "docs"
TOP_K = 3
BATCH_SIZE = 64

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_pdfs(pdf_dir: str) -> List[Document]:
    """Load every PDF under pdf_dir, one Document per page."""
    pdf_paths = sorted(Path(pdf_dir).glob("**/*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found under '{pdf_dir}'.")

    log.info("Found %d PDF(s) in %s", len(pdf_paths), pdf_dir)

    pages = []
    for path in pdf_paths:
        loaded = PyPDFLoader(str(path)).load()
        for i, page in enumerate(loaded):
            page.metadata["source_file"] = path.name
            page.metadata["page"] = i + 1  # human-readable, 1-indexed
        pages.extend(loaded)
        log.info("  %s -> %d page(s)", path.name, len(loaded))

    log.info("Total pages loaded: %d", len(pages))
    return pages


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split pages into overlapping chunks so we don't lose context at page breaks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    log.info("Split %d page(s) into %d chunk(s)", len(documents), len(chunks))
    return chunks


def build_vector_store(chunks: List[Document], embedder: SentenceTransformer):
    """Embed chunks in batches and store them in a persistent Chroma collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)


    if collection.count() >= len(chunks):
        log.info("Collection already has %d chunks indexed, skipping ingestion", collection.count())
        return collection

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [str(i) for i in range(len(chunks))]

    log.info("Embedding %d chunks (batch size %d)...", len(chunks), BATCH_SIZE)
    embeddings = embedder.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False).tolist()

    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    log.info("Indexed %d chunks", len(chunks))
    return collection


def retrieve_context(question: str, collection, embedder: SentenceTransformer, k: int = TOP_K) -> List[dict]:
    """Return the top-k chunks most relevant to the question, each with its source file and page."""
    q_embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[q_embedding], n_results=k)

    if not results["documents"] or not results["documents"][0]:
        return []

    chunks = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": text,
            "source_file": meta.get("source_file", "unknown"),
            "page": meta.get("page", "?"),
        })
    return chunks


def format_context(chunks: List[dict]) -> str:
    """Turn retrieved chunks into a numbered context block the LLM can cite by [n]."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        tag = f"[{i}] {chunk['source_file']}, page {chunk['page']}"
        parts.append(f"{tag}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Run: export GROQ_API_KEY='your-key-here'")
    return Groq(api_key=api_key)


def ask_llm(question: str, context_chunks: List[dict], client: Groq) -> str:
    """Ask the LLM to answer using only the retrieved context, citing chunks by [n]."""
    context = format_context(context_chunks) if context_chunks else "No relevant context found."

    prompt = f"""Answer the question using only the context below.
Cite the source of each claim using its [n] tag, e.g. "RAG retrieves relevant chunks before generation [1]."
If the answer isn't in the context, say you don't know — do not guess.

Context:
{context}

Question: {question}
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def print_sources(context_chunks: List[dict]) -> None:
    """Print the [n] -> file/page mapping so every citation in the answer is verifiable."""
    if not context_chunks:
        return
    print("\nSources:")
    for i, chunk in enumerate(context_chunks, start=1):
        print(f"  [{i}] {chunk['source_file']}, page {chunk['page']}")


def evaluate(eval_set_path: str, collection, embedder: SentenceTransformer) -> None:
    """Check retrieval quality: for each labeled question, did the expected source file
    actually show up in the top-k retrieved chunks?"""
    with open(eval_set_path) as f:
        eval_set = json.load(f)

    hits = 0
    for item in eval_set:
        question = item["question"]
        expected_source = item["expected_source"]

        retrieved = retrieve_context(question, collection, embedder)
        sources = {c["source_file"] for c in retrieved}
        hit = expected_source in sources
        hits += int(hit)

        print(f"{'HIT ' if hit else 'MISS'} - {question}  (expected: {expected_source}, got: {sources or 'nothing'})")

    print(f"\nRetrieval hit rate: {hits}/{len(eval_set)} ({100 * hits / len(eval_set):.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Ask questions over a folder of PDFs (RAG).")
    parser.add_argument("--data", default="./data", help="Folder containing PDF files")
    parser.add_argument("--question", default=None, help="Question to ask (skip this for interactive mode)")
    parser.add_argument("--eval", default=None, help="Path to a labeled eval_dataset.json to check retrieval hit rate")
    args = parser.parse_args()

    try:
        client = get_groq_client()
        embedder = SentenceTransformer(EMBED_MODEL)

        documents = load_pdfs(args.data)
        chunks = chunk_documents(documents)
        collection = build_vector_store(chunks, embedder)

        if args.eval:
            evaluate(args.eval, collection, embedder)
            return

        if args.question:
            questions = [args.question]
        else:
            print("Type a question and hit Enter (Ctrl+C to quit).")
            questions = iter(lambda: input("\nQuestion: "), "")

        for question in questions:
            if not question.strip():
                continue
            context = retrieve_context(question, collection, embedder)
            print(f"\nAnswer: {ask_llm(question, context, client)}")
            print_sources(context)

    except (FileNotFoundError, RuntimeError) as e:
        log.error(e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()

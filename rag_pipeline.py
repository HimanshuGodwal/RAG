import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Configuration --------------------------------------------------------


PDF_DIRECTORY = "pdf_data"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 3

# Documents pre-loaded from PDF_DIRECTORY belong to everyone.
# Documents uploaded through the web app get their own random session_id
# instead, so one person's uploads don't leak into another person's answers.
SHARED_SESSION = "shared"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError("Please set GROQ_API_KEY environment variable.")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

db_client = PersistentClient(path=CHROMA_PATH)
collection = db_client.get_or_create_collection(
    COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def load_pdf_documents(pdf_directory: str):
    pdf_files = list(Path(pdf_directory).glob("**/*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{pdf_directory}'.")

    documents = []

    print(f"\nFound {len(pdf_files)} PDF(s)\n")

    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        for page_number, page in enumerate(pages, start=1):
            page.metadata.update(
                {
                    "source": pdf.name,
                    "page": page_number,
                }
            )

        documents.extend(pages)

        print(f"✓ Loaded {pdf.name} ({len(pages)} pages)")

    print(f"\nTotal Pages : {len(documents)}\n")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    print(f"✓ Created {len(chunks)} chunks\n")

    return chunks


def build_vector_database(chunks) -> None:

    # Only skip files that are already indexed — not the whole batch just
    # because the collection has something in it from a previous run or upload.
    existing_sources = {
        m.get("source") for m in collection.get(include=["metadatas"])["metadatas"]
    }

    chunks = [c for c in chunks if c.metadata.get("source") not in existing_sources]

    if not chunks:
        print("✓ All PDFs in the folder are already indexed. Nothing new to add.\n")
        return

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    start_id = collection.count()

    for offset, chunk in enumerate(chunks):

        documents.append(chunk.page_content)

        embeddings.append(
            embedding_model.encode(chunk.page_content).tolist()
        )

        metadata = chunk.metadata.copy()
        metadata["chunk_id"] = start_id + offset
        metadata["session_id"] = SHARED_SESSION

        metadatas.append(metadata)
        ids.append(str(start_id + offset))

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"✓ Stored {len(documents)} embeddings in ChromaDB\n")


def add_pdf_to_index(file_path: str, session_id: str) -> int:
    """Loads one PDF, chunks it, embeds it, and adds it to the existing collection,
    tagged with the uploader's session_id so it's only visible to that session."""

    filename = Path(file_path).name

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    for page_number, page in enumerate(pages, start=1):
        page.metadata.update(
            {
                "source": filename,
                "page": page_number,
            }
        )

    chunks = split_documents(pages)

    start_id = collection.count()

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for offset, chunk in enumerate(chunks):
        documents.append(chunk.page_content)
        embeddings.append(embedding_model.encode(chunk.page_content).tolist())

        metadata = chunk.metadata.copy()
        metadata["chunk_id"] = start_id + offset
        metadata["session_id"] = session_id
        metadatas.append(metadata)

        ids.append(str(start_id + offset))

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"✓ Indexed {filename} ({len(chunks)} chunks) for session {session_id}\n")

    return len(chunks)


def retrieve_documents(query: str, session_id: str, top_k: int = TOP_K) -> Dict[str, Any]:

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where={"session_id": {"$in": [session_id, SHARED_SESSION]}},
    )

    return results


def build_prompt(question: str, retrieved_documents: Dict[str, Any]) -> str:

    context = "\n\n".join(retrieved_documents["documents"][0])

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer cannot be found inside the context,
reply with:

"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def build_sources(retrieved_documents: Dict[str, Any]) -> list:
    """Turns Chroma's raw query result into a simple list of
    {source, page, match} the frontend can display under each answer."""

    metadatas = retrieved_documents["metadatas"][0]
    distances = retrieved_documents["distances"][0]

    sources = []
    for metadata, distance in zip(metadatas, distances):
        # cosine distance -> similarity percentage (0-100)
        match = round((1 - distance) * 100, 1)
        sources.append(
            {
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", "?"),
                "match": max(0.0, min(100.0, match)),
            }
        )

    return sources


def generate_answer(question: str, session_id: str) -> dict:

    retrieved_documents = retrieve_documents(question, session_id)

    prompt = build_prompt(question, retrieved_documents)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = response.choices[0].message.content

    except Exception as error:
        answer = f"LLM Error: {error}"

    return {
        "answer": answer,
        "sources": build_sources(retrieved_documents),
    }


def main():

    try:

        documents = load_pdf_documents(PDF_DIRECTORY)

        chunks = split_documents(documents)

        build_vector_database(chunks)

        print("=" * 60)
        print("Simple RAG Pipeline Ready")
        print("=" * 60)

        while True:

            question = input("\nAsk a question ('exit' to quit): ")

            if question.lower() == "exit":
                print("Goodbye!")
                break

            result = generate_answer(question, SHARED_SESSION)

            print("\nAnswer:\n")
            print(result["answer"])

            if result["sources"]:
                print("\nSources:")
                for s in result["sources"]:
                    print(f"  - {s['source']} (page {s['page']}) — {s['match']}% match")

    except Exception as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()

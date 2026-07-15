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
collection = db_client.get_or_create_collection(COLLECTION_NAME)


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

    if collection.count() > 0:
        print("✓ Existing ChromaDB detected. Skipping embedding.\n")
        return

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for index, chunk in enumerate(chunks):

        documents.append(chunk.page_content)

        embeddings.append(
            embedding_model.encode(chunk.page_content).tolist()
        )

        metadata = chunk.metadata.copy()
        metadata["chunk_id"] = index

        metadatas.append(metadata)
        ids.append(str(index))

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"✓ Stored {len(documents)} embeddings in ChromaDB\n")


def retrieve_documents(query: str, top_k: int = TOP_K) -> Dict[str, Any]:

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
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


def generate_answer(question: str) -> str:

    retrieved_documents = retrieve_documents(question)

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

        return response.choices[0].message.content

    except Exception as error:
        return f"LLM Error: {error}"


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

            answer = generate_answer(question)

            print("\nAnswer:\n")
            print(answer)

    except Exception as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()

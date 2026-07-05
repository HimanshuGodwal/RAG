# 📚 Retrieval-Augmented Generation (RAG) Pipeline

A beginner-friendly implementation of a Retrieval-Augmented Generation (RAG) pipeline built while learning modern LLM application development.

This project demonstrates the complete RAG workflow—from loading PDF documents to generating context-aware answers using semantic search and a Large Language Model.

---

## 🚀 Features

- Load one or multiple PDF documents
- Split documents into semantic chunks
- Generate embeddings using Sentence Transformers
- Store embeddings in ChromaDB
- Perform semantic similarity search
- Retrieve relevant document chunks
- Generate answers using Groq's Llama 3.1 model

---

## 🛠 Tech Stack

- Python
- LangChain
- ChromaDB
- Sentence Transformers
- PyMuPDF
- LangChain Text Splitters
- Groq API
- Llama 3.1
- Jupyter Notebook

---

## 📂 Pipeline

```
PDF Documents
      │
      ▼
Document Loader
      │
      ▼
Text Splitter
      │
      ▼
Sentence Embeddings
      │
      ▼
ChromaDB Vector Store
      │
      ▼
Similarity Search
      │
      ▼
Retrieved Context
      │
      ▼
Llama 3.1 (Groq)
      │
      ▼
Final Answer
```

---

## Workflow

1. Load PDF documents.
2. Split documents into manageable chunks.
3. Generate embeddings using `all-MiniLM-L6-v2`.
4. Store embeddings in ChromaDB.
5. Convert the user's query into an embedding.
6. Retrieve the most relevant chunks.
7. Send the retrieved context to the LLM.
8. Generate a context-aware answer.

---

## Project Structure

```
RAG/
│
├── RAG.ipynb
├── data/
│   └── *.pdf
└── README.md
```

---

## Example Query

```
Question:
What is CRC?
```

The system retrieves the most relevant document chunks before generating an answer with Llama 3.1.

---

## Concepts Practiced

- Retrieval-Augmented Generation (RAG)
- Document Loading
- Text Chunking
- Vector Embeddings
- Semantic Search
- ChromaDB
- Prompt Engineering
- Context Injection
- Large Language Models

---

## Future Improvements

- Convert notebook into a Python package
- Build a FastAPI backend
- Create a Streamlit web interface
- Store vectors persistently
- Support multiple embedding models
- Add metadata filtering
- Add conversation memory
- Dockerize the application
- Evaluate retrieval quality
- Deploy on the cloud

---

## Learning Objective

This project was built as part of my journey to learn Retrieval-Augmented Generation (RAG) and modern LLM application development. The primary goal was to understand each stage of the RAG pipeline before moving on to larger production-ready AI applications.

---

## Author

**Himanshu Godwal**

AI & Machine Learning Student  
Interested in LLMs, RAG, Computer Vision, and AI Engineering.

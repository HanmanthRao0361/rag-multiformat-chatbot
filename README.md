# rag-multiformat-chatbot
A Retrieval-Augmented Generation (RAG) chatbot that supports multiple document formats (PDF, text, etc.) for intelligent question-answering using vector search and LLMs.
# RAG Multiformat Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to query and chat with multiple document formats (PDF, text, etc.) using vector search and Large Language Models (LLMs).

---

## 📌 Overview

This project implements a modular RAG pipeline where documents are ingested, converted into embeddings, stored in a FAISS vector database, and queried using an LLM to generate context-aware responses.

The application is designed with clean separation of concerns and can be extended to support additional document formats and vector stores.

---

## 🧠 Architecture

User Query
|
v
LLM Prompt
|
v
Retriever (FAISS Vector Store)
|
v
Relevant Chunks


---

## ✨ Features

- Supports multiple document formats (PDF, text)
- Document ingestion and chunking
- Vector-based semantic search using FAISS
- Context-aware responses using LLMs
- Modular and extensible code structure
- Environment-based configuration (secure API keys)

---

## 🛠️ Tech Stack

- **Programming Language:** Python
- **LLM Framework:** LangChain
- **Vector Store:** FAISS
- **Backend:** FastAPI
- **Embeddings:** HuggingFace / OpenAI (configurable)
- **Environment Management:** python-dotenv

---

## 📂 Project Structure

rag-multiformat-chatbot/
│
├── data_loader.py # Handles document loading and preprocessing
├── rag_chain.py # RAG pipeline and chain logic
├── newapp.py # Application entry point (API / chatbot logic)
├── faiss_index/ # Generated FAISS index (ignored in Git)
├── .gitignore
├── README.md
└── requirements.txt

---


v
LLM Response

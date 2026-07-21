# Intermediate RAG System with Pinecone

A Streamlit web app that answers questions from uploaded PDF documents using Pinecone Vector Database and Groq LLM.

## Features
- Upload PDF files (Up to 20MB)
- Extract text and store embeddings in Pinecone
- Get accurate answers strictly from PDF content
- View source pages and context excerpts
- Adjustable Chunk Size, Top-K, and Threshold settings from UI

## Setup & Run

1. **Install required libraries:**
```bash
pip install -r requirements.txt
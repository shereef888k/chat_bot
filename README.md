# AI Chatbot (RAG System)

## Overview
This project is an AI chatbot that answers user queries using Retrieval-Augmented Generation (RAG). It retrieves relevant information from custom data and generates context-aware responses.

## Features
- Semantic search using FAISS
- Context-based answers using embeddings
- Backend API using Flask
- Supports custom data (PDF, text, etc.)

## Tech Stack
- Python
- Flask
- FAISS
- Sentence Transformers

## How it works
1. User sends a query
2. Query is converted to embeddings
3. FAISS searches similar data
4. Relevant context is returned as answer

## Setup
```bash
git clone https://github.com/shereef888k/chat_bot
cd chat_bot
pip install -r requirements.txt
python app.py

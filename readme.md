# RAG Local Knowledge Base Question and Answer System
## packages:
- pypdf2
- dashscope
- langchain
- langchain-openai
- langchain-community
- faiss-cpu
- fast-api

## instruction:
    A RAG knowledge-based question-answering system developed using the Langchain framework. 
    It supports multi-text vector retrieval, model backend switching, and FastAPI services.

## run
1. Install the required packages:
2. start the FastAPI service:
    uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
3. run main.py
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_pipeline import chat_with_query

app = FastAPI(title="RAG Demo API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer:str
    references:list
    usage_metadata:dict

@app.post("/query", response_model=QueryResponse)
async def query_api(request:QueryRequest):
   try:
       result = chat_with_query(request.query)
       return result
   except Exception as e:
       logging.error(f"Query processing failed: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))
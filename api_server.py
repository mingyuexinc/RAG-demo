import logging
import os
import shutil
import uuid
from enum import unique

from fastapi import FastAPI, HTTPException, File, UploadFile
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import Config
from data_loader import data_loader
from rag_pipeline import chat_with_query
from retriever import generate_query_variants, retrieve_with_score
from vector_store import get_or_create_vector_database

app = FastAPI(title="RAG Demo", version="1.0.1")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer:str
    references:list
    usage_metadata:dict

class UploadResponse(BaseModel):
    message:str
    filename:str
    file_id:str

class SearchResponse(BaseModel):
    query:str
    retrieved_documents:list
    similarity_score:list[float]


@app.post("/query", response_model=QueryResponse)
async def query_api(request:QueryRequest):
   try:
       # call rag_pipeline
       result = chat_with_query(request.query)
       return result
   except Exception as e:
       logging.error(f"Query processing failed: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload",response_model=UploadResponse)
async def upload_document(file:UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_extension = file.filename.split(".")[-1]
        # type check
        if file_extension not in Config.FILE_SUFFIX:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        upload_dir = Config.FILE_LOAD_PATH
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        file_path = os.path.join(upload_dir, f"{file_id}.{file_extension}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # chunks
        chunks = data_loader(file_path)
        # create vector db
        get_or_create_vector_database(chunks)

        return UploadResponse(
            message="File uploaded successfully",
            filename=file.filename,
            file_id=file_id
        )

    except Exception as e:
        logging.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search",response_model=SearchResponse)
async def search(query:str):
    vector_db = get_or_create_vector_database()
    multi_query = generate_query_variants(query)
    all_results = []

    for query in multi_query:
        results = retrieve_with_score(vector_db, query)
        all_results.extend(results)

    unique = {}
    for doc,score in all_results:
        key = doc.page_content
        if key not in unique or score > unique[key][1]:
            unique[key] = (doc,score)

    final_docs = sorted(unique.values(), key=lambda x: x[1], reverse=True)[:5]

    return SearchResponse(
        query=query,
        retrieved_documents=[doc.page_content for doc, score in final_docs],
        similarity_score=[score for doc, score in final_docs]
    )



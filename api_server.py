import logging
import os
import shutil
import uuid

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel

from config import Config
from data_loader import data_loader
from rag_pipeline import chat_with_query
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



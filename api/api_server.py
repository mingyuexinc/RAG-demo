import logging
import os
import shutil
import uuid

from fastapi import FastAPI, HTTPException, File, UploadFile

from config.app_config import AppConfig
from core.container import AppContainer
from data_loader import data_loader
from core.planner import TaskPlanner
from result.response_api import QueryResponse, QueryRequest, UploadResponse
from result.response_generator import process_tool_result

from vector_store import get_or_create_vector_database

app = FastAPI(title="RAG Agent", version="1.0.2")



@app.post("/tool/execute", response_model=QueryResponse)
async def execute_tool(request:QueryRequest):
   try:
       planner = TaskPlanner()

       # init agent
       doc_agent = AppContainer.get_doc_agent()

       plan = planner.analyze_task(request.query)

       result = doc_agent.execute(plan)

       return process_tool_result(result,doc_agent,request)

   except Exception as e:
       logging.error(f"Query processing failed: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload",response_model=UploadResponse)
async def upload_document(file:UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_extension = file.filename.split(".")[-1]
        # type check
        if file_extension not in AppConfig.vector.FILE_SUFFIX:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        upload_dir = AppConfig.vector.FILE_LOAD_PATH
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

# @app.get("/search",response_model=SearchResponse)
# async def search(query:str):
#     vector_db = get_or_create_vector_database()
#     multi_query = generate_query_variants(query)
#     all_results = []
#
#     for query in multi_query:
#         results = retrieve_with_score(vector_db, query)
#         all_results.extend(results)
#
#     unique = {}
#     for doc,score in all_results:
#         key = doc.page_content
#         if key not in unique or score > unique[key][1]:
#             unique[key] = (doc,score)
#
#     final_docs = sorted(unique.values(), key=lambda x: x[1], reverse=True)[:5]
#
#     return SearchResponse(
#         query=query,
#         retrieved_documents=[doc.page_content for doc, score in final_docs],
#         similarity_score=[score for doc, score in final_docs]
#     )



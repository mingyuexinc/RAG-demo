from typing import Dict, Any, Optional
from core.agent import DocAgent
from core.executor import ExecutionResult, TaskType
from fastapi import HTTPException
from pydantic import BaseModel

from prompts.prompt_manager import PromptManager


class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    task_type:TaskType
    answer:Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    usage_metadata: Optional[list] = None

class UploadResponse(BaseModel):
    message:str
    filename:str
    file_id:str

class SearchResponse(BaseModel):
    query:str
    retrieved_documents:list
    similarity_score:list[float]

class ResponseGenerator:
    def __init__(self, prompt_manager: PromptManager):
        self.pm = prompt_manager

    @staticmethod
    def generate(
        result: ExecutionResult,
        doc_agent: DocAgent,
        query: str
    ) -> Dict[str, Any]:

        if result.task_type == "knowledge_qa":
            return ResponseGenerator._knowledge_qa(result, doc_agent, query)

        elif result.task_type == "flowchart_generation":
            return ResponseGenerator._flowchart(result)

        else:
            pass

        raise ValueError(f"Unsupported task_type: {result.task_type}")


    def _knowledge_qa(
            self,
            result: ExecutionResult,
            doc_agent: DocAgent,
            query: str
    ) -> Dict[str, Any]:
        prompt = self.pm.render(
            "templates/knowledge_qa_template.txt",
            context=result.final_context,
            query = query
        )

        answer = doc_agent.generate_response(prompt)

        documents = result.tool_results.get("knowledge_search", {}).get("documents", [])
        references = [doc.get("content", "") for doc in documents]
        metadata = [doc.get("metadata", {}) for doc in documents]

        return {
            "task_type":result.task_type,
            "answer": answer,
            "references": references,
            "usage_metadata": metadata
        }

    @staticmethod
    def _flowchart(result: ExecutionResult) -> Dict[str, Any]:
        chart_result = result.tool_results.get("chart_gen", {})

        return {
            "task_type":result.task_type,
            "answer": "已根据制度文档生成流程图。",
            "payload": {
                "chart_url": chart_result.get("chart_url"),
                "chart_code": chart_result.get("chart_code")
            }
        }


def process_tool_result(
    result: ExecutionResult,
    doc_agent: DocAgent,
    request: QueryRequest
):
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)

    response_data = ResponseGenerator.generate(
        result=result,
        doc_agent=doc_agent,
        query=request.query
    )

    return QueryResponse(**response_data)

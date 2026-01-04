from typing import Dict, Any, Optional
from core.agent import DocAgent
from core.executor import ExecutionResult, TaskType
from fastapi import HTTPException
from prompts.prompt_manager import PromptManager
from result.response_api import QueryRequest, QueryResponse


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
        knowledge_search_result = result.tool_results["knowledge_search"]
        documents = knowledge_search_result["data"]["documents"]
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

        chart_result = result.tool_results["chart_gen"]
        chart_data = chart_result["data"]

        return {
            "task_type":result.task_type,
            "answer": "已根据制度文档生成流程图。",
            "payload": {
                "chart_url": chart_data.get("chart_url"),
                "chart_code": chart_data.get("chart_code")
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

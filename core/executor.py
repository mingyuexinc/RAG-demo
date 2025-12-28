from dataclasses import dataclass

from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Literal

TaskType = Literal[
    "knowledge_qa",
    "flowchart_generation",
    "summary"
]

TASK_TOOL_CONSTRAINTS = {
    "knowledge_qa": ["knowledge_search"],
    "flowchart_generation": ["knowledge_search", "summarizer", "chart_gen"],
    "summary": ["knowledge_search", "summarizer"],
}


class ExecutionResult(BaseModel):
    success: bool
    task_type: TaskType
    executed_tools: List[str]
    tool_results: Dict[str, Any]
    error: Optional[str] = None


class ExecutionContext:
    def __init__(self):
        self.data = {}
        self.meta = {}

    def set(self,key:str,value:Any):
        self.data[key] = value

    def get(self,key:str) -> Any:
        return self.data.get(key)

    def get_by_path(self, path: str):
        if path in self.data:
            return self.data[path]

        for output_key, value in self.data.items():
            if path.startswith(output_key + "."):
                sub_path = path[len(output_key) + 1:]
                return self._resolve_subpath(value, sub_path)

        return None

    def _resolve_subpath(self, value: Any, sub_path: str):
        parts = sub_path.split(".")
        current = value
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

@dataclass
class ExecutionPlan:
    task_type: TaskType
    need_tools: bool
    tools: List[str]
    tool_params: Dict[str, Dict[str, Any]]

    def validate(self, available_tools: List[str]):
        if self.task_type not in TASK_TOOL_CONSTRAINTS:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        allowed_tools = TASK_TOOL_CONSTRAINTS[self.task_type]

        for tool in self.tools:
            if tool not in available_tools:
                raise ValueError(f"Unknown tool: {tool}")

            if tool not in allowed_tools:
                raise ValueError(
                    f"Tool {tool} not allowed for task_type {self.task_type}"
                )

        for tool in self.tools:
            if tool not in self.tool_params:
                raise ValueError(f"Missing params for tool: {tool}")
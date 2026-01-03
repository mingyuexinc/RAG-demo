from copy import deepcopy
from typing import Dict, Any

from core.executor import ExecutionResult, ExecutionPlan, ExecutionContext
from model import ModelManager
from tools.base_tool import BaseTool


class DocAgent:
    def __init__(self,tools:Dict[str,BaseTool],max_steps = 5,max_retries = 3,max_content_size=100):
        self.tools = tools
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.max_content_size = max_content_size
        self.session_context = {}

    def execute(self,plan:ExecutionPlan)-> ExecutionResult:
        executed_tools = []
        tool_results = {}
        context = ExecutionContext(max_size=self.max_content_size)
        step_count = 0

        try:
            for tool_name in plan.tools:
                if step_count >= self.max_steps:
                    raise ValueError("Exceeded maximum execution steps: {self.max_step}")

                state_snapshot = {
                    "executed_tools": executed_tools.copy(),
                    "tool_results": deepcopy(tool_results),
                    "context": deepcopy(context)
                }

                retry_count = 0
                success = False

                while retry_count < self.max_retries and not success:
                    try:
                        if tool_name not in self.tools:
                            raise ValueError(f"Unknown tool: {tool_name}")
                        tool = self.tools[tool_name]

                        raw_params = plan.tool_params.get(tool_name, {})
                        resolved_params = self._resolve_params(raw_params, context)
                        if tool_name == "knowledge_search":
                            set_tool = True
                        else:
                            set_tool = False
                        result = tool.run(resolved_params, context, set_tool)

                        executed_tools.append(tool_name)
                        tool_results[tool_name] = result
                        success = result.get("success")
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= self.max_retries:
                            raise ValueError(f"Failed to execute tool: {tool_name} after {self.max_retries} retries: {str(e)}")
                        else:
                            executed_tools = state_snapshot["executed_tools"].copy()
                            tool_results = state_snapshot["tool_results"].deepcopy()
                            context = state_snapshot["context"].deepcopy()
                step_count += 1

            return ExecutionResult(
                success = True,
                task_type=plan.task_type,
                executed_tools = executed_tools,
                tool_results = tool_results,
            )

        except Exception as e:
            return ExecutionResult(
                success = False,
                task_type=plan.task_type,
                executed_tools = executed_tools,
                tool_results = tool_results,
                error = str(e)
            )

    def _resolve_params(self,params:Dict[str,Any],context:ExecutionContext) -> Dict[str,Any]:
        resolved = {}
        for key,value in params.items():
            if isinstance(value, str) and "." in value:
                resolved_value = context.get_by_path(value)
                if resolved_value is None:
                    raise ValueError(f"Missing dependency: {value}")
                resolved[key] = resolved_value

            else:
                resolved[key] = value
        return resolved


    def generate_response(self, prompt:str) -> str:
        model_manager = ModelManager()
        chat_llm_model = model_manager.create_model_instance()
        response = chat_llm_model.invoke(input=prompt)
        return response.content.strip()






from typing import Dict, Any

from prompts.prompt_manager import PromptManager
from .executor import ExecutionResult, ExecutionPlan, ExecutionContext
from model import ModelManager
from tools.base_tool import BaseTool


class DocAgent:
    def __init__(self,tools:Dict[str,BaseTool]):
        self.tools = tools

    def execute(self,plan:ExecutionPlan)-> ExecutionResult:
        executed_tools = []
        tool_results = {}
        context = ExecutionContext()
        try:
            for tool_name in plan.tools:
                if tool_name not in self.tools:
                    raise ValueError(f"Unknown tool: {tool_name}")
                tool = self.tools[tool_name]

                raw_params = plan.tool_params.get(tool_name,{})
                resolved_params = self._resolve_params(raw_params, context, tool)
                if tool_name == "knowledge_search":
                    set_tool = True
                else:
                    set_tool = False
                result = tool.run(resolved_params,context,set_tool)

                executed_tools.append(tool_name)
                tool_results[tool_name] = result

            return ExecutionResult(
                success = True,
                task_type=plan.task_type,
                executed_tools = executed_tools,
                tool_results = tool_results,
            )
        except Exception as e:
            return ExecutionResult(
                success = False,
                executed_tools = executed_tools,
                tool_results = tool_results,
                error = str(e)
            )

    def _resolve_params(self,params:Dict[str,Any],context:ExecutionContext,tool:BaseTool) -> Dict[str,Any]:
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






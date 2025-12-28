from core.executor import ExecutionContext
from model import ModelManager
from prompts.prompt_manager import PromptManager
from tools.base_tool import BaseTool


class SummaryTool(BaseTool):
    name = "summarizer"
    input_keys = ["documents"]
    output_key = "summarizer.result"

    def __init__(self):
        super().__init__(name=self.name)
        self.prompt = PromptManager()

    def execute(self,context:ExecutionContext) -> str:
        try:
            documents = context.get("knowledge_search.result")
            documents = documents.get("documents")
            if not documents or not isinstance(documents, list):
                raise ValueError("No valid documents found in context")
            content = "\n".join([doc.get("content", "") for doc in documents])
            if not content:
                raise ValueError("No content found in documents")
            summarizer_prompt = self.prompt.render(
                "templates/summarizer_template.txt",
                content=content,
                max_length=500
            )

            model_manager = ModelManager()
            chat_model = model_manager.create_model_instance()
            response = chat_model.invoke(input=summarizer_prompt)
            summary_text = response.content.strip()

            context.set(self.output_key, summary_text)
            return summary_text
        except Exception as e:
            raise ValueError(f"Failed to summarize content: {str(e)}")




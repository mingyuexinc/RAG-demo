import os

from langchain_openai import ChatOpenAI

from config import Config


class ModelManager:
    @staticmethod
    def create_model_instance() -> ChatOpenAI:
        return ChatOpenAI(
            api_key=os.getenv(Config.DASHSCOPE_API_KEY),
            base_url=Config.INFERENCE_MODEL_URL,
            model=Config.INFERENCE_MODEL_NAME
        )


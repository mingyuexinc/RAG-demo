import os

from langchain_openai import ChatOpenAI

from config.app_config import AppConfig


class ModelManager:
    @staticmethod
    def create_model_instance() -> ChatOpenAI:
        return ChatOpenAI(
            api_key=os.getenv(AppConfig.model.DASHSCOPE_API_KEY),
            base_url=AppConfig.model.INFERENCE_MODEL_URL,
            model=AppConfig.model.INFERENCE_MODEL_NAME
        )


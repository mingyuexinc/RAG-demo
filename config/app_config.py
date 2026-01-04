from config.agent_config import AgentConfig
from config.executor_config import ExecutorConfig
from config.model_config import ModelConfig
from config.vector_config import VectorConfig


class AppConfig:
    model = ModelConfig
    agent = AgentConfig
    executor = ExecutorConfig()
    vector = VectorConfig()



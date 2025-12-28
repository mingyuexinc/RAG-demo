
class Config:
    """
    Config class for the application.
    """

    # model
    DASHSCOPE_API_KEY = "DASHSCOPE_API_KEY"
    INFERENCE_MODEL_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    INFERENCE_MODEL_NAME = "qwen-plus"

    # vector_db
    FILE_LOAD_PATH = "./assets/upload"
    VECTOR_DB_SAVE_PATH = "./vector_db"

    # embedding mode
    EMBEDDING_MODEL_NAME = "text-embedding-v2"

    # text splitter
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128

    # support file type
    FILE_SUFFIX = ["pdf", "doc", "docx", "txt"]

    JSON_TASK_SCHEMA = """
    {
      "task_type": "knowledge_qa | flowchart_generation | summary",
      "need_tools": true,
      "tools": ["knowledge_search", "summarizer", "chart_gen"],
      "tool_params": {
         "knowledge_search": {
            "query": "string"
         },
         "summarizer": {
            "documents": "knowledge_search.result.documents"
         },
         "chart_gen": {
            "summarized_text": "summarizer.result"
         }
      }
    }
    """


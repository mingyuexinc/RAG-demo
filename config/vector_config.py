from config.base_config import BaseConfig


class VectorConfig(BaseConfig):
    FILE_LOAD_PATH = "./assets/upload"
    VECTOR_DB_SAVE_PATH = "./vector_db"

    # text splitter
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128

    # support file type
    FILE_SUFFIX = ["pdf", "doc", "docx", "txt"]
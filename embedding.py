from langchain_community.embeddings import DashScopeEmbeddings
from config import Config

def build_embedding():
    # deploy embedding model
    embeddings = DashScopeEmbeddings(
        model=Config.EMBEDDING_MODEL_NAME
    )
    return embeddings
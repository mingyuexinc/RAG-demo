import logging
import os
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from data_loader import data_loader
from embedding import build_embedding
from config import Config


def get_vector_database(chunks:List[str],embeddings:DashScopeEmbeddings, save_path:str = None)->FAISS:

    # create knowledge database
    knowledge_database = FAISS.from_texts(chunks, embeddings)
    logging.debug("create knowledge database from chunks...")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        knowledge_database.save_local(save_path)
        logging.debug(f"knowledge database saved to {save_path}...")

    return knowledge_database

def load_vector_database(load_path:str,embeddings = None) -> FAISS:
    """
    :param load_path:path for saving vector database
    :param embeddings:optional,if None,a new DashScope embedding instance will be created
    :return:storage objects based on FAISS
    """

    # load FAISS
    knowledge_database = FAISS.load_local(load_path, embeddings,allow_dangerous_deserialization=True)
    logging.debug(f"vector database loaded from {load_path}")

    return knowledge_database

def create_vector_database() -> FAISS:
    loader_path = Config.FILE_LOAD_PATH
    chunks_test = data_loader(loader_path)
    embedding_model = build_embedding()
    save_dir = Config.VECTOR_DB_SAVE_PATH
    if not os.path.exists(save_dir) or os.path.getsize(save_dir) == 0:
        vector_database = get_vector_database(chunks_test,embedding_model,save_path=save_dir)
    else:
        vector_database = load_vector_database(save_dir)
    return vector_database


# if __name__ == "__main__":
#     create_vector_database()

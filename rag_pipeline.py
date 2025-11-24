from langchain_classic.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import get_usage_metadata_callback

from config import Config
from vector_store import get_or_create_vector_database

import os



def rag_pipeline(query:str, vectorstore:FAISS):

    # similarity search execution
    # docs = vectorstore.similarity_search(query)

    # initialize chatLLM
    chatLLM = ChatOpenAI(
        api_key=os.getenv(Config.DASHSCOPE_API_KEY),
        base_url=Config.INFERENCE_MODEL_URL,
        model=Config.INFERENCE_MODEL_NAME
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever = vectorstore.as_retriever(),
        llm = chatLLM
    )

    references = retriever.invoke(query)

    # build context
    context = "\n".join([doc.page_content for doc in references])

    # create prompt
    prompt = f"""
    基于以下文档内容回答问题：

    文档内容：
    {context}

    问题：{query}

    请根据文档内容回答问题，如果文档中没有相关信息，请说明无法从文档中找到答案。
    """

    references_data = []
    for i,reference in enumerate(references):
        references_data.append({
            "content":reference.page_content
        })

    # answer
    with get_usage_metadata_callback() as cost:
        response = chatLLM.invoke(input=prompt)

    # response
    response_data = {
        "answer":response.content,
        "references":references_data,
        "usage_metadata":cost.usage_metadata
    }

    return response_data

def chat_with_query(query:str):
    vector_db = get_or_create_vector_database()
    answer = rag_pipeline(query,vector_db)
    return answer

# if __name__ == "__main__":
#     question = "客户经理的考核标准是什么?"
#     chat_with_query(question)
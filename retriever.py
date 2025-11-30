import os

from langchain_openai import ChatOpenAI

from config import Config


def generate_query_variants(question: str):
    prompt = f"""
    请为下面的问题生成 3 个语义不同但相关的查询：
    {question}
    """
    chatLLM = ChatOpenAI(
        api_key=os.getenv(Config.DASHSCOPE_API_KEY),
        base_url=Config.INFERENCE_MODEL_URL,
        model=Config.INFERENCE_MODEL_NAME
    )

    resp = chatLLM.invoke(prompt)
    return [q.strip("-• ") for q in resp.content.split("\n") if len(q.strip())>0]

def retrieve_with_score(db, query, k=5):
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "score_threshold": 0.0, "return_score": True}
    )
    return db.similarity_search_with_score(query, retriever=retriever)
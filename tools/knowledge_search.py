from langchain_community.vectorstores import FAISS
from core.executor import ExecutionContext
from model import ModelManager
from tools.base_tool import BaseTool


class KnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    input_keys = ["query"]
    output_key = "knowledge_search.result"

    def __init__(self, vector_store: FAISS):
        super().__init__(name=self.name)
        self.vector_store = vector_store

    def execute(self,context:ExecutionContext):
        query = context.get("query")
        docs = self.retrieve_with_score(self.vector_store, query, 5)
        result = {
            "documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in docs
            ]
        }
        context.set(self.output_key, result)


    def generate_query_variants(self,question: str):
        prompt = f"""
        请为下面的问题生成 3 个语义不同但相关的查询：
        {question}
        """
        model_manager = ModelManager()
        chat_llm_model = model_manager.create_model_instance()
        resp = chat_llm_model.invoke(prompt)
        return [q.strip("-• ") for q in resp.content.split("\n") if len(q.strip())>0]

    def retrieve_with_score(self,db, query, k=5):
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "score_threshold": 0.0, "return_score": True}
        )
        return db.similarity_search_with_score(query, retriever=retriever)
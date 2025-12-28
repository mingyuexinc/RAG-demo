from core.agent import DocAgent
from tools.chart_gen import ChartGenTool
from tools.knowledge_search import KnowledgeSearchTool
from tools.summarizer import SummaryTool
from vector_store import get_or_create_vector_database


class AppContainer:
    _doc_agent: DocAgent = None

    @classmethod
    def get_doc_agent(cls) -> DocAgent:
        if cls._doc_agent is None:
            vector_db = get_or_create_vector_database()

            tools = {
                "knowledge_search": KnowledgeSearchTool(vector_db),
                "summarizer": SummaryTool(),
                "chart_gen": ChartGenTool(),
            }

            cls._doc_agent = DocAgent(tools)

        return cls._doc_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

# Web search tool
def web_search_tool():
    web_search_tool = TavilySearchResults(k=3)
    return web_search_tool

def web_search(question):
    docs = web_search_tool().invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return web_results
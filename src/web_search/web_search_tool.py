from langchain_community.tools.tavily_search import TavilySearchResults

# Web search tool
def web_search_tool():
    web_search_tool = TavilySearchResults(k=3)
    return web_search_tool
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults


@tool
def search_online(query: str) -> str:
    """
    `search_online` tool Search the internet for information..
    Input parameters:
    query (string)      :       phrase of given question.

    Output :
    result (string)     :       online search result
    """
    search_tool = DuckDuckGoSearchResults()
    result = search_tool.invoke(query)

    return result
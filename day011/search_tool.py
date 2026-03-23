from langchain_community.tools import DuckDuckGoSearchResults

search_tool = DuckDuckGoSearchResults()
results = search_tool.invoke("What is Resident evil 9 requiem?")

print(results)
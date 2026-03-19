from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=5,
    lang="en"
)

query = "Geopolitical history of USA and Russia from the perspective of Iran."
documents = retriever.invoke(query)
print(documents)
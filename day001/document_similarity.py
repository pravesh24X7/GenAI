from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

query="New Delhi is the capital of India, situtated in the Northern region."
docs = [
    "Paris is the capital of France and It is famous for Eiffel Tower",
    "Germany is the coutry which fought 2 consecutive world wars.",
    "Delhi is situtated between Haryana and Punjab region in India.",
]

query_embedding = embeddings.embed_query(query)
docs_embedding = embeddings.embed_documents(docs)

similarities = cosine_similarity([query_embedding], docs_embedding)[0]     # output shape is (1, n) that's why [0]
print("similarity scores: ", similarities)

most_similar_index = similarities.argmax()
print("Most relevant document: ", docs[most_similar_index])
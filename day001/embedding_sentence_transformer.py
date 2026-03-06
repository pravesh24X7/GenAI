from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text = "What is the capital of India?"
vector = embeddings.embed_query(text)

docs = [
    "New Delhi is the capital of India",
    "situated in the northern part between Punjab and Haryana"
]

matrix = embeddings.embed_documents(docs)

print(len(vector))
print(len(matrix))
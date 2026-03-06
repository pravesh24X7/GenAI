from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",
                 dimensions=32)
vector = embedding.embed_query("What is the captial of India?")

docs = [
    "New Delhi is the capital of India",
    "It is situated in the Northern part of India",
    "specifically between Punjab and Haryana region."
]

matrix = embedding.embed_documents(docs)

print(vector.shape)
print(matrix.shape)

# above code didn't worked because OpenAI requires subscription.
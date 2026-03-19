from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
documents = [
    Document(page_content="Lion is the king of Jungle."),
    Document(page_content="Mango is the king of Fruits."),
    Document(page_content="Rainbow has seven colors in patter VIBGYOR."),
    Document(page_content="Langchain is the open source framework used to develop Agentic-AI apps."),
]

vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model,
)

retreiver = vector_store.as_retriever(
    search_type="mmr",      # this enables MMR
    search_kwargs={"k": 2,  # top_K results 
                   "lambda_mult": 1,    # lambda_mult, relevance-diversity balance, if 1 then works like simple similarity search, if 0 returns very diverse results, optimal value lies between 0-1.
                   }
)
query = "king of?"
results = retreiver.invoke(query)
print(results)
from langchain_community.vectorstores import Chroma
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

vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db",
    collection_name="sample-coll1",
)

# upload docs to vector store using add_documents fxn
vector_store.add_documents(documents=documents)

# create retriever of vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
query = "king of?"

results = retriever.invoke(query)
print(results)
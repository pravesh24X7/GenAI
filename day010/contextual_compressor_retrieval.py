from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(page_content="Lion is the king of Jungle."),
    Document(page_content="Mango is the king of Fruits."),
    Document(page_content="Rainbow has seven colors in patter VIBGYOR."),
    Document(page_content="Langchain is the open source framework used to develop Agentic-AI apps."),
]

embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
documents = [
    Document(page_content="Lion is the king of Jungle."),
    Document(page_content="Mango is the king of Fruits."),
    Document(page_content="Rainbow has seven colors in patter VIBGYOR."),
    Document(page_content="Langchain is the open source framework used to develop Agentic-AI apps."),
]

vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

base_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "lambda_mult": 0.5}
)
query = "king of?"

# setup compressor model
llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.5,
                     model_kwargs={})
compressor = LLMChainExtractor.from_llm(llm_model)

# create compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

results = compression_retriever.invoke(query)
print(results)
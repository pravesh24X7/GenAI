from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path="./testfile.pdf")
documents = loader.lazy_load()

splitter = CharacterTextSplitter(chunk_size=100,    
                                chunk_overlap=0,    # describes the overlapping region between chunks, for RAG based apps (10-20) is good no.
                                separator="")
results = splitter.split_documents(documents=documents)

print(len(results))
print(results[:10])
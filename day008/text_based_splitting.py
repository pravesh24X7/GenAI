from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader(file_path="./testfile.pdf")
documents = loader.lazy_load()

splitter = RecursiveCharacterTextSplitter(chunk_overlap=0,
                                          chunk_size=50,
                                          )
results = splitter.split_documents(documents=documents)

print(len(results))
print(results[:5])
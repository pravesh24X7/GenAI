from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path=r"C:\Users\papsr\Desktop\lit. survey",
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
documents = loader.lazy_load()

for doc in documents:
    print(doc.metadata)
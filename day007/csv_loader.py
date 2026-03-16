from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("./testfile.csv")
documents = loader.load()

print(len(documents))   # for each row in .csv file a separate document object is created.
for doc in documents:
    print(doc.page_content)
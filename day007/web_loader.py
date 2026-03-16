from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv

load_dotenv()
loader = WebBaseLoader(web_paths=("https://www.pcgamer.com/news/",))
documents = loader.lazy_load()

content = ""

total_docs = 0
for doc in documents:
    content += doc.page_content.strip()
    total_docs += 1

llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.5,
                     model_kwargs={})
prompt_template = PromptTemplate(template="List all the major news from the below given content.\n{news}",
                                 input_variables=["news"],
                                 validate_template=True)
parser = StrOutputParser()

chain = prompt_template | llm_model | parser
response = chain.invoke({"news": content})

print(response)
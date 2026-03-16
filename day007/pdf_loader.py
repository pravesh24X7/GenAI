from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.5,
                     model_kwargs={})

parser = StrOutputParser()

prompt_template = PromptTemplate(template="Rate the below given resume on the scale of 1 to 10.\n{resume}",
                                 validate_template=True,
                                 input_variables=["resume"])

loader = PyPDFLoader("./testing_resume.pdf")
documents = loader.load()

chain = prompt_template | llm_model | parser
response = chain.invoke({
    "resume": documents[0].page_content
})

print(response)
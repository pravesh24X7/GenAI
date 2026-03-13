from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.45,
                     model_kwargs={})

prompt_template = load_prompt("./task1_template.json")
chain = prompt_template | llm_model | parser

final_response = chain.invoke({
    "topic": "Blackhole"
})
print(final_response)
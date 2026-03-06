from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


llm_model = OpenAI(model="gpt-3.5-turbo-instruct")
model_response = llm_model.invoke("What is the capital of India?")      #   contains query which model will answer.

print(model_response)

# above code is not working because OpenAI models required subscription 
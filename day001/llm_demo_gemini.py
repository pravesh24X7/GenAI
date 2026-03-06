from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("GOOGLE_API_KEY"))


llm_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model_response = llm_model.invoke("What is the capital of India?")

print(model_response)

# above code will not work, similar to OpenAI, Google also works on subscriptions
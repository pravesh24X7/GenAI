import os
from langchain_groq import ChatGroq 
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("GROQ_API_KEY"))



llm_model = ChatGroq(model="llama-3.1-8b-instant",
                     temperature=0, # controls the randomness of language model's output. it affects how creative or deterministic the response are. lower value (deterministic), higher value (creative or random)
                    #  max_completion_tokens=100, # at max. how many tokens (roughly words) needed. (OpenAI, google) works like this
                    model_kwargs={"max_completion_tokens": 50}  # Groq works in this way.
                     )
model_response = llm_model.invoke("What is the capital of India?")
print(model_response.content)

print(model_response.response_metadata)
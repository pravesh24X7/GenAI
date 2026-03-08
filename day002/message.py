from langchain_core.messages import (SystemMessage,
                                     HumanMessage,
                                     AIMessage)
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# create LLM object
llm_model = ChatGroq(model="llama-3.1-8b-instant",
                     temperature=0.2,
                     model_kwargs={})

messages = [
    SystemMessage(content="For all the questions you've to answer only in one word."),
    HumanMessage(content="What is the capital of India?"),
]

messages.append(
    AIMessage(llm_model.invoke(messages).content)
)
print(messages)
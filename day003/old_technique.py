# Old technique
from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
from langchain_core.messages import (SystemMessage,
                                     AIMessage,
                                     HumanMessage)
from dotenv import load_dotenv

load_dotenv()

llm_model = ChatGroq(model="llama-3.1-8b-instant",
                     temperature=0.23,
                     model_kwargs={})

# load template
prompt_template = load_prompt("./sports.json")
history = [
    SystemMessage(content="Return the response strictly in valid JSON format with key=`answer_(question_number)`"),
    HumanMessage(content=prompt_template.invoke(
        {
            "sports": "football"
        }
    ).to_string())  # if we use the .format function, we don't need to use .to_string() fxn
]

model_response = llm_model.invoke(history)
history.append(AIMessage(content=model_response.content))

print(history)
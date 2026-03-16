from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import (SystemMessage,
                                     AIMessage,
                                     HumanMessage)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.5,
                     model_kwargs={})

prompt_template = PromptTemplate(template="{query}",
                                 validate_template=True,
                                 input_variables=["query"])

loader = TextLoader("./imaginary_ironman.txt", encoding="UTF-8")
documents = loader.load()

parser = StrOutputParser()
chain = prompt_template | llm_model | parser

history = [
    SystemMessage(content="Behave like a chatbot answer the user's query. \nMax. output word limit is 50 words." + " " + documents[0].page_content),
]

while True:
    user_input = input("Human : ", ).strip()
    if user_input.lower() == "exit":
        break

    msg = history.append(HumanMessage(content=user_input))
    response = chain.invoke({
        "query": history
    })

    print("AI : ", response)
    history.append(AIMessage(content=response))
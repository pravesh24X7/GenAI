from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import (AIMessage,
                                     SystemMessage,
                                     HumanMessage)
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

history = []

llm_model = ChatGroq(model="llama-3.1-8b-instant",
                     temperature=0.45,
                     model_kwargs={})

parser = StrOutputParser()

system_msg = SystemMessage(content="You're an expert of {topic}.")
prompt = ChatPromptTemplate.from_messages([
    system_msg,
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template(
        "{task}"
    )
])

chain = prompt | llm_model | parser
response1 = chain.invoke({
    "topic": "Physics",
    "task": "Provide 200 words description about Nuclear Fusion.",
    "history": history,
})

print("Response 1: ", response1)
history.append(AIMessage(content=response1))

sec_chain = prompt | llm_model | parser
response2 = sec_chain.invoke({
    "topic": "Physics",
    "task": "Generate a 50 words summary of previous task.",
    "history": history
})

print("Response 2: ", response2)
history.append(AIMessage(content=response2))

# for visualizing the above chain, both chains are some where identical
chain.get_graph().print_ascii()
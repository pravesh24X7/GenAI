from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub
from langchain_groq import ChatGroq
from basic_search_tool import search_online
from dotenv import load_dotenv

load_dotenv()

llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.55,
                     model_kwargs={})

# pull the ReAct (means Reasoning + Acting agent) pompt from hub
prompt = hub.pull("hwchase17/react")

# create ReAct agent manually with pulled prompt.
agent = create_react_agent(
    llm=llm_model,
    prompt=prompt,
    tools=[search_online]
)

# wrap it with AgentExecutor, which is responsible for performing actions.
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_online],
    verbose=True
)

# invoke
response = agent_executor.invoke({
    "input": "Convert 5 USD to INR, using latest conversion rates of 2026"
})

print(response)
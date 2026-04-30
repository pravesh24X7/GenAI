from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import TypedDict
from dotenv import load_dotenv


load_dotenv()

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str


llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     model_kwargs={},
                     temperature=0.5)

parser = StrOutputParser()

prompt = PromptTemplate(template="You're helpful assistant, Solve the given user query.\n\nQuery: {query}",
                        input_variables=['query'],
                        validate_template=True)

chain = prompt | llm_model | parser


def generate_joke(state: JokeState) -> dict:
    joke = chain.invoke({
        "query": f"Generate a funny joke on topic: {state['topic']}."
    })
    return {'joke': joke}


def explain_joke(state: JokeState) -> dict:
    explanation = chain.invoke({
        'query': f"Explain the below given joke on topic: {state['topic']}\n\nJoke: {state['joke']}."
    })
    return {'explanation': explanation}


graph = StateGraph(JokeState)

graph.add_node('generate_joke', generate_joke)
graph.add_node('explain_joke', explain_joke)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'explain_joke')
graph.add_edge('explain_joke', END)

# before compiling the graph into a workflow, configure the checkpointer
checkpointer = InMemorySaver()  # use to store in RAM

workflow = graph.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

initial_state = {
    'topic': "Indian Politics"
}
final_state = workflow.invoke(initial_state, config=config)

print(final_state)

print("All State Values:")
# list all the values of the state.
# print(list(workflow.get_state_history(config=config)))

config2 = {
    "configurable": {
        "thread_id": "2"
    }
}

initial_state = {
    'topic': "Bollywood"
}
final_state = workflow.invoke(initial_state, config=config2)

print(final_state)

print("All State Values:")
# print(list(workflow.get_state_history(config=config2)))

# to print the required values, just pass the thread_id
final_state_config2 = workflow.get_state(config=config2)
print(final_state_config2)

# img = workflow.get_graph().draw_mermaid_png()
# with open("./img001.png", "wb") as f:
#     f.write(img)
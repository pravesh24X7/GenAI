from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Literal, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver     # stores into RAM
from dotenv import load_dotenv


load_dotenv()


class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]


llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.5,
                     model_kwargs={})


def chat_node(state: ChatState) -> dict:

    messages = state['messages']
    response = llm_model.invoke(messages)
    return {'messages': [response]}


checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)

initial_state, final_state = None, None
# messages = []   # why maintain a separate state when we've configured the reducer function in the state, because reducers function do not work in multiple .invoke() calls, it only works in single execution. Each time we call invoke function state is reset and then execution begins from starting.

# alternative solution using MemorySaver
thread_id = '1'

while True:
    query = input('[ HUMAN ] : ').strip()

    if query.lower() == "exit":
        break
    # messages.append(HumanMessage(content=query))
    initial_state = {
        'messages': [HumanMessage(content=query)]
    }

    config = {'configurable': {'thread_id': thread_id}}

    final_state = workflow.invoke(initial_state, config=config)
    print(f"[ AI ] : {final_state['messages'][-1].content}")


print(final_state)


img = workflow.get_graph().draw_mermaid_png()
with open("./chatbot001.png", "wb") as f:
    f.write(img)
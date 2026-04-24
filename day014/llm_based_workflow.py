from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import TypedDict


# load the environment variables
load_dotenv()

# invnoke llm
llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                      temperature=0.5,
                      model_kwargs={})

# parser object
parser = StrOutputParser()

# defining prompt
prompt = PromptTemplate(template="You're an helpful assistant, Provide answers to user query. \n\nQuery: {query}",
                        validate_template=True,
                        input_variables=["query"])

# execution chain
chain = prompt | llm_model | parser


# defining state
class LLMChatState(TypedDict):
    query: str
    response: str


# functions
def get_query_response(state: LLMChatState) -> LLMChatState:
    
    response = chain.invoke({
        'query': state["query"],
    }) 

    state["response"] = response
    return state


# defining graph
graph = StateGraph(LLMChatState)

# add nodes
graph.add_node("llm", get_query_response)

# add edges
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

# compile graph
workflow = graph.compile()

# begin execution
initial_state = {
    "query": "What do you mean by Photosynthesis?",
}

final_state = workflow.invoke(initial_state)
print(final_state)

# save graph
img_file = workflow.get_graph().draw_mermaid_png()

with open("./img002.png", "wb") as f:
    f.write(img_file)
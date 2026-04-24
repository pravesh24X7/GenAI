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
class LLMState(TypedDict):
    topic: str
    outline: str
    blog: str
    rating: float


# function
def get_outline(state: LLMState) -> LLMState:

    response = chain.invoke({
        "query": f"Generate a blog outline for '{state['topic']}'"
    })

    state["outline"] = response
    return state


def generate_blog(state: LLMState) -> LLMState:

    response = chain.invoke({
        "query": f"{state['topic']}{state['outline']}"
    })

    state["blog"] = response
    return state


def get_rating(state: LLMState) -> LLMState:

    rating = chain.invoke({
        "query": f"Rate the below given blog on the scale of 1 to 5. Output only the number. Do not provide any other information.\n\n{state['blog']}"
    })

    state['rating'] = float(rating)
    return state


# defining graph
graph = StateGraph(LLMState)

# add nodes
graph.add_node("blog_outline", get_outline)
graph.add_node("write_blog", generate_blog)
graph.add_node("rate_blog", get_rating)

# add edges
graph.add_edge(START, "blog_outline")
graph.add_edge("blog_outline", "write_blog")
graph.add_edge("write_blog", "rate_blog")
graph.add_edge("rate_blog", END)

# compile graph
workflow = graph.compile()

# begin execution
initial_state = {
    "topic": "Importance of AI in Medical Science",
}

final_state = workflow.invoke(initial_state)
print(final_state)

# save graph
img_file = workflow.get_graph().draw_mermaid_png()

with open("./img003.png", "wb") as f:
    f.write(img_file)
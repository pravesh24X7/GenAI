import operator

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
from dotenv import load_dotenv


load_dotenv()

prompt = PromptTemplate(template="You're a Strict IAS officer recruiter. Query: {query}. \n\nEssay: {essay}.",
                        input_variables=["query", "essay"],
                        validate_template=True)

parser = StrOutputParser()


class EvaluationSchema(BaseModel):
    feedback: str = Field(default="", description="Detailed feedback of essay.")
    score: float = Field(default=1, description="Score in the range of 1 to 5", ge=1, le=5)

llm_model000 = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                        temperature=0.5,
                        model_kwargs={})

llm_model001 = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                        temperature=0.5,
                        model_kwargs={}).with_structured_output(EvaluationSchema)

llm_model002 = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                        temperature=0.5,
                        model_kwargs={}).with_structured_output(EvaluationSchema)

llm_model003 = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                        temperature=0.5,
                        model_kwargs={}).with_structured_output(EvaluationSchema)


# defining state
class AnswerState(TypedDict):
    essay: str
    cot_feedback: str
    doa_feedback: str
    lang_feedback: str
    overrall_feedback: str
    individual_scores: Annotated[list[float], operator.add]
    average_score: float



# functions
def evaluate_cot(state: AnswerState) -> dict:
    user_essay = state["essay"]

    # execution chain
    chain = prompt | llm_model001
    response = chain.invoke({
        "query": "Judge the essay on the basis of `Chain of Thoughts` and Provide a detailed feedback along with score in the range of 1 to 5",
        "essay": user_essay
    })

    return {
        "cot_feedback": response.feedback,
        "individual_scores": [float(response.score)]
    }


def evaluate_doa(state: AnswerState) -> dict:
    user_essay = state["essay"]

    # execution chain
    chain = prompt | llm_model002
    response = chain.invoke({
        "query": "Evaluate the essay on the basis of `Depth of Analysis` and Provide a detailed feedback along with score in the range of 1 to 5",
        "essay": user_essay
    })

    return {
        "doa_feedback": response.feedback,
        "individual_scores": [float(response.score)]
    }


def evaluate_language(state: AnswerState) -> dict:
    user_essay = state["essay"]

    # execution chain
    chain = prompt | llm_model003
    response = chain.invoke({
        "query": "Provide rating for the given essay on the basis of `language` and Provide a detailed feedback along with score in the range of 1 to 5.",
        "essay": user_essay
    })

    return {
        "lang_feedback": response.feedback,
        "individual_scores": [float(response.score)]
    }


def final_evaluation(state: AnswerState) -> AnswerState:
    prompt_0 = PromptTemplate(template="Write a Summarized feedback on the basis of below gievn context: {context}",
                              validate_template=True,
                              input_variables=["context"])

    chain = prompt_0 | llm_model000 | parser
    response = chain.invoke({
        "context": f"{state['cot_feedback']}\n\n{state['doa_feedback']}\n\n{state['lang_feedback']}"
    })

    state["overrall_feedback"] = response
    state["average_score"] = sum(state["individual_scores"]) / len(state["individual_scores"])

    return state


# declare graph
graph = StateGraph(AnswerState)


# add nodes
graph.add_node("cot", evaluate_cot)
graph.add_node("doa", evaluate_doa)
graph.add_node("language", evaluate_language)
graph.add_node("summary", final_evaluation)

# add edges
graph.add_edge(START, "cot")
graph.add_edge(START, "doa")
graph.add_edge(START, "language")
graph.add_edge(["cot", "doa", "language"], "summary")   # means: summary waits until ALL 3 finish
graph.add_edge("summary", END)

# compile graph
workflow = graph.compile()

# begin execution
initial_state = {
    "essay": """
Democracy in India is the largest in the world, shaped by its vast cultural and social diversity. Since independence, the country has followed a representative system guided by the Constitution of India, which guarantees fundamental rights and promotes equality. Elections allow citizens from both rural and urban areas to influence governance, making participation a key strength. However, challenges such as economic inequality, regional disparities, and the influence of money and media in politics affect democratic functioning. Despite these issues, institutions like the judiciary and Election Commission help maintain balance. Indian democracy continues to adapt, reflecting both the complexity and resilience of its society.
    """
}

final_state = workflow.invoke(initial_state)

print(final_state)
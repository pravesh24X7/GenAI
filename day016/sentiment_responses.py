
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, TypedDict
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


load_dotenv()


class QuerySentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the given query.")


template = """
    You've to perform the text classification, return the response only in word, Do not provide any explaination.
        classification categories are: ["positive", "negative"]
    \n\n
    Query: {query}
"""

prompt = PromptTemplate(template=template,
                        input_variables=["query"],
                        validate_template=True)

llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.5,
                     model_kwargs={}).with_structured_output(QuerySentiment)

base_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.5,
                     model_kwargs={})


chain = prompt | llm_model


class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str


def find_sentiment(state: ReviewState) -> dict:
    
    sentiment = chain.invoke({
        "query": state["review"]
    }).sentiment

    return {'sentiment': sentiment}


def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    if state['sentiment'] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"



def positive_response(state: ReviewState) -> dict:
    response = llm_model.invoke(f"Write a warm thank you message for the given review.:\n\nReview: {state['review']}")
    return {"response": response.content}


def negative_response(state: ReviewState) -> dict:
    
    query = f"you're a supportive assistant. \nThe user had a {state['diagnosis']['issue_type']} issue, sounded: {state['diagnosis']['tone']} and marked urgency as {state['diagnosis']['urgency']}.\nWrite an empathetic, helpful resolution message."

    response = base_model.invoke(query).content
    return {"response": response}



def run_diagnosis(
        state: ReviewState
        ) -> dict:
    

    class Diagnose(BaseModel):
        issue_type: str = Field(description="Description of the review")
        tone: str = Field(description="Tone in which user is providing the review")
        urgency: Literal["low", "medium", "high"] = Field(description="level of urgency of the review")

    json_parser = JsonOutputParser(pydantic_object=Diagnose)
    
    query = f"Diagnose the given review : \n\nReview: {state['review']}.\n\nReturn response in JSON format, where keys are `issue_type`, `tone` and `urgency`.\n\nInstructions: {json_parser.get_format_instructions()}"

    response = base_model.invoke(query)

    return {"diagnosis": dict(json_parser.parse(response.content))}

graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("positive_response", positive_response)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)
graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)


workflow = graph.compile()

initial_state= {
    "review": "This app is like the Tejaswi yadav of software industry."
}
final_state = workflow.invoke(initial_state)
print(final_state)


img = workflow.get_graph().draw_mermaid_png()
with open("./img002.png", mode="wb") as f:
    f.write(img)

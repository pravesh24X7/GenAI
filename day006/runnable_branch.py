from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableBranch, RunnablePick
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

class Sentiment(BaseModel):
    sentiment : Literal["positive", "negative"] = Field(description="an attitude, thought, or judgment prompted by feeling, emotion, or opinion rather than strict reason.")

llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.5,
                     model_kwargs={})
parser1 = JsonOutputParser(pydantic_object=Sentiment)
parser2 = StrOutputParser()
sentiment_prompt = load_prompt("./sentiment_template.json")

sentiment_chain = {
    "comment": RunnablePick("comment"),
    "sentiment": sentiment_prompt | llm_model | parser1
}

positive_sentiment_prompt = PromptTemplate(template="Appreciate for the good sentiment in just 10 words.\n {comment}.\n{sentiment}",
                                           validate_template=True,
                                           input_variables=["comment", "sentiment"])
negative_sentiment_prompt = PromptTemplate(template="Criticize the people mentioned in the following comment in just 10 words.\n{comment}.\n{sentiment}",
                                           validate_template=True,
                                           input_variables=["comment", "sentiment"])

parallel_chain = RunnableParallel({
    "comment": RunnablePick(keys=["comment"]),
    "sentiment": RunnablePick(keys=["sentiment"]),
    "result": RunnableBranch(
        ((lambda x: x["sentiment"]["sentiment"] == "positive"), positive_sentiment_prompt | llm_model | parser2),
        ((lambda x: x["sentiment"]["sentiment"] == "negative"), negative_sentiment_prompt | llm_model | parser2),
        (RunnableLambda(
            lambda x: "Sorry, Can't do that now. !!!"
        ))
    )
})

final_chain = sentiment_chain | parallel_chain
final_response = final_chain.invoke({
    "comment": "I love my country, because peoples have zero civic sense.",
    "format_instruction": parser1.get_format_instructions(),
})

print(final_response)
final_chain.get_graph().print_ascii()
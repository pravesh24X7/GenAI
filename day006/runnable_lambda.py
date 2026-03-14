import math
from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

class Sentiment(BaseModel):
    sentiment : Literal["positive", "negative"] = Field(description="an attitude, thought, or judgment prompted by feeling, emotion, or opinion rather than strict reason.")

llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.5,
                     model_kwargs={})
parser = JsonOutputParser()

sentiment_prompt = load_prompt("./sentiment_template.json")
        
sentiment_chain = sentiment_prompt | llm_model | parser
parallel_chain = RunnableParallel({
    "sentiment": RunnablePassthrough(),
    "result": RunnableLambda(
        lambda x: math.sqrt(4) if x["sentiment"] == "positive" else math.log10(1000)
    )
})

final_chain = sentiment_chain | parallel_chain
final_response = final_chain.invoke({
    "comment": "I love my country, because peoples have zero civic sense.",
    "format_instruction": parser.get_format_instructions(),
})

print(final_response)
final_chain.get_graph().print_ascii()
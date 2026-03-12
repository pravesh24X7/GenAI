from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

class Feedbacks(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="contains positive and negative feedback classification"
    )

parser = PydanticOutputParser(pydantic_object=Feedbacks)
str_output_parser = StrOutputParser()

llm_model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.2
)

prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback as positive or negative.

{feedback}

{format_instruction}
""",
    input_variables=["feedback", "format_instruction"]
)

classifier_chain = prompt1 | llm_model | parser


positive_feedback_prompt = PromptTemplate(
    template="""
Write an appropriate response to this positive feedback:

{feedback}
""",
    input_variables=["feedback"]
)

negative_feedback_prompt = PromptTemplate(
    template="""
Write an appropriate response to this negative feedback:

{feedback}
""",
    input_variables=["feedback"]
)


# combine feedback and classification
combined_chain = RunnableParallel(
    sentiment=classifier_chain,
    feedback=lambda x: x["feedback"]
)


branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].sentiment == "positive",
        positive_feedback_prompt | llm_model | str_output_parser
    ),
    (
        lambda x: x["sentiment"].sentiment == "negative",
        negative_feedback_prompt | llm_model | str_output_parser
    ),
    RunnableLambda(lambda x: "Could not determine sentiment")
)

final_chain = combined_chain | branch_chain

response = final_chain.invoke({
    "feedback": "This is a very good product, I love using this. My friend recommended this and I will recommend it to all.",
    "format_instruction": parser.get_format_instructions()
})

print(response)
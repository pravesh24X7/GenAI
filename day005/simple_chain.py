from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (SystemMessage,
                                     AIMessage,
                                     HumanMessage)
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

template="""
    You're expert of {domain}.\n
    Explain {topic} in 5 lines.

    {format_instructions}
"""
global_msg = SystemMessage(content="For all the question, return the output in JSON format where key is `response`")

prompt = ChatPromptTemplate.from_messages([
    global_msg,
    ("human", template)
])

class Article(BaseModel):
    response: str = Field(description="contains answer of the question")

parser =  JsonOutputParser(pydantic_object=Article)

llm_model = ChatGroq(model="llama-3.1-8b-instant",
                     temperature=0.45,
                     model_kwargs={})

sequential_chain = prompt | llm_model | parser
final_response = sequential_chain.invoke({
    "domain": "Physics",
    "topic": "Gravitation",
    "format_instructions": parser.get_format_instructions(),
})

print(final_response)

# to visualize the above execution chain
sequential_chain.get_graph().print_ascii()
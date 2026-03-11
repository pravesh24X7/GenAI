from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import PydanticOutputParser
from typing import Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Person(BaseModel):
    name: str = Field(description="Name of the person", default="Unknown")
    age: int = Field(gt=17, default=18, description="Age of the person")
    city: str = Field(description="City name where the person lives.", default="unknown")

parser = PydanticOutputParser(pydantic_object=Person)

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct:featherless-ai",
                          task="text-generation",
                          )
llm_model = ChatHuggingFace(llm=llm)

prompt = load_prompt("./pydantic_template.json")
chain = prompt | llm_model | parser

final_response = chain.invoke({
    "place": "USA",
    "format_instruction": parser.get_format_instructions()
})
print(final_response)
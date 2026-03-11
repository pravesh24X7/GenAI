from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct:featherless-ai",
                          task="text-generation",
                          )
llm_model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

prompt = load_prompt("./JsonOutputParserTemplate.json")

chain = prompt | llm_model | parser

final_response = chain.invoke({"format_instruction": parser.get_format_instructions()})

print(final_response)
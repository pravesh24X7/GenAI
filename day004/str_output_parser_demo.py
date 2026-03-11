from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct:featherless-ai",
                          task="text-generation",
                          )
llm_model = ChatHuggingFace(llm=llm)

prompt1 = load_prompt("./StrOutputParserTemplate1.json")
prompt2 = load_prompt("./StrOutputParserTemplate2.json")

# In efficient way of doing the right things.
# final_prompt1 = prompt1.invoke({
#     "topic": "Blackhole"
# }).to_string()

# model_response = llm_model.invoke(final_prompt1)

# final_prompt2 = prompt2.invoke({
#     "text": model_response.content
# }).to_string()

# final_model_reponse = llm_model.invoke(final_prompt2)

# optimal way
# first create an object of OutputParser
parser = StrOutputParser()

# create a chain of execution
chain = prompt1 | llm_model | parser | prompt2 | llm_model | parser

# now simply call the .invoke function on the above chain, you'll get your final response directly.
final_response = chain.invoke({
    "topic": "Blackhole"
})
print(final_response)
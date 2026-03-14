from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Tweet(BaseModel):
    tweet : str = Field(description="A phrase which can be posted on Social Media.")

class Post(BaseModel):
    post : str = Field(description="A formal post that describes the context.")

tweet_template = load_prompt("./tweet_template.json")
linkedIn_template = load_prompt("./linkedIn_template.json")

llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                      temperature=0.35,
                      model_kwargs={})

parser = JsonOutputParser()

parallel_chain = RunnableParallel({
    "tweet": tweet_template | llm_model | parser,
    "post": linkedIn_template | llm_model | parser
})
final_response = parallel_chain.invoke({
    "topic": "Blackhole",
    "format_instruction": parser.get_format_instructions()
})
print(final_response)

# visualize the chain
parallel_chain.get_graph().print_ascii()
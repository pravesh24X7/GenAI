from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
from langchain_core.messages import (SystemMessage,
                                     AIMessage,
                                     HumanMessage)
from dotenv import load_dotenv
from typed_dict import QA

load_dotenv()

# create LLM
llm_model = ChatGroq(model="llama-3.1-8b-instant",
                     temperature=0.23,
                     model_kwargs={})

# wrap llm_model with SportGenre schema
structured_llm_model = llm_model.with_structured_output(QA)

# load template
prompt_template = load_prompt("./sports.json")

history = [
    SystemMessage(content="Answer the questions and return the response in JSON format with key 'answer' containing a list of answers."),
    HumanMessage(content=prompt_template.format(sports="football")),
]

response: QA = structured_llm_model.invoke(history)
print(response["answer"])

# the above code will not work because the LLM model which we're using, do NOT reliably support tool/function calling, which is what with_structured_output() uses internally.
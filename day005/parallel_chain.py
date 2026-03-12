from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
parser = StrOutputParser()
notes_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You're an expert of {field}."
    ),
    HumanMessagePromptTemplate.from_template(
        "Generate short notes on {topic}."
    )
])

quiz_prompt = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template(
        "You're an expert of {field}."
    ),
    HumanMessagePromptTemplate.from_template(
        "Generate 5 basic quiz question on {topic}."
    )
])

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct:featherless-ai",
                          task="text-generation",
                          )
llm_model1 = ChatHuggingFace(llm=llm)

llm_model2 = ChatGroq(model="llama-3.1-8b-instant",
                      temperature=0.2,
                      model_kwargs={})

llm_model3 = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                      temperature=0.2,
                      model_kwargs={})

chain1 = notes_prompt | llm_model1 | parser
chain2 = quiz_prompt | llm_model2 | parser

parallel_chain = RunnableParallel({
    "response1": chain1,
    "response2": chain2,
})

final_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("Merge the given phrase and question. Remember don't provide answer to any questions."),
    HumanMessagePromptTemplate.from_template("Phrase: {response1}\nQuestions: {response2}")
])

final_chain = parallel_chain | final_prompt | llm_model3 | parser
final_response = final_chain.invoke({
    "field": "Biology",
    "topic": "Reproduction",
})

print(final_response)

# visualize the chain
final_chain.get_graph().print_ascii()
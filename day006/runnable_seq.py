from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(template="Write a joke on {topic}",
                        validate_template=True,
                        input_variables=["topic"])

llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.5,
                     model_kwargs={})
parser = StrOutputParser()

prompt2 = PromptTemplate(template="Explain the given joke.\n{joke}")

# alternate way of doing the same is: prompt | llm_model | parser | prompt2 | llm_model | parser
# chain = RunnableSequence(prompt, llm_model, parser, prompt2, llm_model, parser)
# final_response = chain.invoke({
#     "topic": "Smartphones"
# })

# print(final_response)

joke_generator = RunnableSequence(prompt, llm_model, parser)
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, llm_model, parser)
})

final_chain = joke_generator | parallel_chain
final_response = final_chain.invoke({
    "topic": "Blackhole"
})
print(final_response)
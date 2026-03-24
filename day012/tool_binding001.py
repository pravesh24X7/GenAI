from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from my_custom_tool import custom_addition_fxn
from dotenv import load_dotenv

load_dotenv()

llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.35,
                     model_kwargs={})

# bind the custom_addition_fxn tool to llm model
llm_model_with_tools = llm_model.bind_tools([custom_addition_fxn])     # if there are other tools as well, append them in same list seprated by ","

prompt_template = PromptTemplate(
    template="""
Use the Custom tool to perform addition operation.
Extract integers properly and pass them as integers (not strings).

{query}
""",
    input_variables=["query"],
    validate_template=True,
)

chain = prompt_template | llm_model_with_tools
response = chain.invoke({
    "query": "Perform custom addition operation\n Value of operands are 3 and 2."
})

# extract tool call
tool_call = response.tool_calls[0]

if tool_call["name"] == "custom_addition_fxn":
    tool_result = custom_addition_fxn.invoke(tool_call)

print(tool_result)      # response is tool message.

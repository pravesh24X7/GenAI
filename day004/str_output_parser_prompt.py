from langchain_core.prompts import PromptTemplate

template1="""
    Write a detailed report on {topic}
"""

template2="""
    Write a 5 line summary on following text.\n
    {text}
"""

prompt_template1 =  PromptTemplate(
    template=template1,
    validate_template=True,
    input_variables=["topic"]
)

prompt_template2 = PromptTemplate(
    template=template2,
    input_variables=["text"],
    validate_template=True
)

prompt_template1.save("StrOutputParserTemplate1.json")
prompt_template2.save("StrOutputParserTemplate2.json")
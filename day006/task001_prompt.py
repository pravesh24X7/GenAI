from langchain_core.prompts import PromptTemplate

template="""
    Task is to write 70 to 100 words on {topic}
"""

prompt_template = PromptTemplate(template=template,
                                validate_template=True,
                                input_variables=["topic"])
prompt_template.save("./task1_template.json")
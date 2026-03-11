from langchain_core.prompts import PromptTemplate

template="""
    Generate the name, age and city of the fictional {place} person. \n
    Return ONLY a JSON object.
    {format_instruction}
"""
prompt_template = PromptTemplate(template=template,
                                 validate_template=True,
                                 input_variables=["place", "format_instruction"])

prompt_template.save("pydantic_template.json")
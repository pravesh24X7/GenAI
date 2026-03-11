from langchain_core.prompts import PromptTemplate

template = """
    Give me name, age and city of a fictional character \n
    {format_instruction}
"""

prompt_template = PromptTemplate(template=template,
                                 validate_template=True,
                                 input_variables=["format_instruction"])

# use the below given technique when prompt and agent is present in the same file, because partial_variable value are not filled while saving the prompt. 
# prompt_template = PromptTemplate(template=template,
#                                  validate_template=True,
#                                  input_variables=[],
#                                  partial_variables={ 
#                                      "format_instruction": parser.get_format_instructions() # get the JSON format inst.
#                                  })

prompt_template.save("JsonOutputParserTemplate.json")
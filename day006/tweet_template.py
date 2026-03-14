from langchain_core.prompts import PromptTemplate

template="""
    Write a tweet on {topic}.
    Return the response in JSON format with key `tweet`.
     \n{format_instruction}
"""
prompt_template = PromptTemplate(template=template,
                                 validate_template=True,
                                 input_variables=["topic", "format_instruction"])
prompt_template.save("./tweet_template.json")
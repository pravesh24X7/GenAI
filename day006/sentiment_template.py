from langchain_core.prompts import PromptTemplate

template="""
    Return the sentiment of below given comment.\n
    {comment}.

    Return the response in JSON format with key `sentiment`
    \n
    {format_instruction}
"""

prompt_template = PromptTemplate(template=template,
                                 validate_template=True,
                                 input_variables=["format_instruction", "comment"])
prompt_template.save("./sentiment_template.json")
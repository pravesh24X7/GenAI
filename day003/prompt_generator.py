from langchain_core.prompts import (PromptTemplate,
                                    )

template = """
    You are an experienced {sports} coach.
    Answer the below given questions from {sports} perspective.

    1. When did first match played?
    2. Which 2 team played the first match?
    3. How many players are there in each team?
"""

prompt_template = PromptTemplate(template=template,
                              validate_template=True,
                              input_variables=["sports"])
prompt_template.save("sports.json")
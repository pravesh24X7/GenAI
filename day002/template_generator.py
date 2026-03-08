from langchain_core.prompts import PromptTemplate

template = """
Summarize the research paper titled {topic}, with following specifications:
Explaination style: {style}
Explaination length: {length}

1. Mathematical details:
    - Include relevant mathematical equations if present in paper.
    - Explain the mathematical concepts using simple, intutive code snippets where applicable.
2. Analogies:
    - Use relatable analogies to simplify complex ideas.
If sufficient infomation is not available in paper, respond with `Insufficient information available` instead of guessing.
Ensure the summary is clear, accurate, and aligned with provided style and length.
"""

# instead of this manual work use PromptTemplate
# final_prompt = template.format(
#     topic=paper_input,
#     style=style_input,
#     length=length
# )

prompt_template = PromptTemplate(
    input_variables=["topic", "style", "length"],
    template=template,
    validate_template=True,
)

prompt_template.save("template.json")
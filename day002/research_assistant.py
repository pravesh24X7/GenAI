from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Here we demonstrate the working of static prompts.

# create LLM object
llm_model = ChatGroq(model="llama-3.1-8b-instant",
                     temperature=0.3,
                     model_kwargs={
                        #  "max_completion_tokens": 100
                     })

# website building part using streamlit
st.header("Research Assistant Application")     # .header is used to add header in website
user_input = st.text_input("Your input question goes here ...")

if st.button("Answer Prompt"):
    model_response = llm_model.invoke(user_input)
    st.text(model_response.content)

"""
In most scenarios, user given prompts doesn't works like they wanted. sometimes a combination of visualization or maths or even code algorithms gives better understanding of current topic.

to handle this, the ideal approach is to create a prompt template that changes according to user needs.
example:

    Summarize the research paper titled {topic}, with following specifications:
    Explaination style: {style}
    Explaination length: {length}

    1. Mathematical details:
        Include relevant mathematical equations if present in paper.
        Explain the mathematical concepts using simple, intutive code snippets where applicable.
    2. Analogies:
        Use relatable analogies to simplify complex ideas.
    If sufficient infomation is not available in paper, respond with `Insufficient information available` instead of guessing.
    Ensure the summary is clear, accurate, and aligned with provided style and length.
"""


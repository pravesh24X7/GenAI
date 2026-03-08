from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# In most scenarios, user given prompts doesn't works like they wanted. sometimes a combination of visualization or maths or even code algorithms gives better understanding of current topic.

# to handle this, the ideal approach is to create a prompt template that changes according to user needs.
# example:

#     Summarize the research paper titled {topic}, with following specifications:
#     Explaination style: {style}
#     Explaination length: {length}

#     1. Mathematical details:
#         Include relevant mathematical equations if present in paper.
#         Explain the mathematical concepts using simple, intutive code snippets where applicable.
#     2. Analogies:
#         Use relatable analogies to simplify complex ideas.
#     If sufficient infomation is not available in paper, respond with `Insufficient information available` instead of guessing.
#     Ensure the summary is clear, accurate, and aligned with provided style and length.


# Create object of LLM model
llm_model = ChatGroq(model="llama-3.1-8b-instant",
                     temperature=0.2,
                     model_kwargs={})

# streamlit code
st.header("Dynamic Research Assistant App")

paper_input = st.selectbox("Select paper name",
                           [
                               "Select",
                               "Attention is all you need",
                               "BERT: Pre-training of Deep Bidirectional Transformer",
                               "GPT-3: Language models are few short learners",
                               "Diffusion models beats GANs on Image Synthesis"
                           ])

style_input = st.selectbox("Select Explaination style",
                           [
                               "Beginner-friendly",
                               "Technical",
                               "Code-oriented",
                               "Math-oriented"
                           ])

length = st.selectbox("Select Explaination length",
                      [
                          "Short (1-2) paragraphs",
                          "Medium (5-8) paragraphs",
                          "Long (10-12) detailed paragraph explainations"
                      ])

# load prompt from template file
prompt_template = load_prompt("template.json")

if st.button("Answer Prompt"):

    # this is beginners approach
    # final_prompt = prompt_template.invoke(
    #     topic=paper_input,
    #     style=style_input,
    #     length=length
    # )
    # model_response = llm_model.invoke(final_prompt)

    # professionals approach, using chains
    chain = prompt_template | llm_model
    model_response = chain.invoke({
        "topic": paper_input,
        "style": style_input,
        "length": length,
    })
    st.write(model_response.content)

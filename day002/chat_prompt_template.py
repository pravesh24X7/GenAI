from langchain_core.messages import (SystemMessage,
                                     HumanMessage,)
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Ineffecient way, placeholders don't get's values
# chat_template =  ChatPromptTemplate([
#     SystemMessage(content="You're a helpful {domain} expert."),
#     HumanMessage(content="Explain {topic} in simple terms.")
# ])

# alternative way
# chat_template = ChatPromptTemplate.from_messages([
#     ("system", "You're a helpful {domain} expert."),
#     ("human", "Explain {topic} in simple terms.")
# ])

# best way of doing the same.
chat_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You're a helpful {domain} expert."
    ),
    HumanMessagePromptTemplate.from_template(
        "Explain {topic} in simple terms."
    )
])

prompt = chat_template.invoke({
    "domain": "Sports",
    "topic": "Cricket"
})

print(prompt)
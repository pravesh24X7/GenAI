from langchain_core.prompts import (MessagesPlaceholder, ChatPromptTemplate)
from langchain_core.prompts import (SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate,)

# chat template
chat_template = ChatPromptTemplate.from_messages(
    SystemMessagePromptTemplate.from_template(
        "you're an expert of {domain}"
    ),
    MessagesPlaceholder(
        variable_name="chat_history"
    ),
    HumanMessagePromptTemplate.from_template(
        "{query}"
    )
)

# load chat history
chat_history = []
with open("filename.txt") as f:
    chat_history.extend(f.readline())

# create prompt
final_prompt = chat_template.invoke({
    "domain": "sports",
    "chat_history": chat_history,
    "query": "explain football in simple terms."
})
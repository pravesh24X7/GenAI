from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
from langchain_core.messages import (SystemMessage,
                                     HumanMessage,
                                     AIMessage)
from dotenv import load_dotenv

load_dotenv()

# create LLM object
llm_model = ChatGroq(model="llama-3.1-8b-instant",
                     temperature=0.2,
                     model_kwargs={})

# the only problem is the model, do not remember the previous chats, to handle this we've to store chat_history and send this with each new prompt so that model will remember and answer accordingly.
chat_history = [
    SystemMessage("For all the questions, you've to answer only in 1 word.")
]

# this will run infinite, until user type `exit`
while True:
    user_input = HumanMessage(input("\n\nYou: ").strip())

    # store user_prompts in chat_history
    chat_history.append(user_input)

    # check if user_input is exit, if so terminate the program.
    if user_input.content == "exit":
        break

    model_response = AIMessage(llm_model.invoke(chat_history).content)
    print("\n\nAI:", model_response.content)

    # along with the model response
    chat_history.append(model_response.content)

# once chatting is done, store this in database or some file wherever, you want.
print(chat_history)

# Instead of sending whole chat history in real world applications, Hybrid memory architecture is implemented.
# User Query
#      │
#      ▼
# Retrieve relevant memory (vector DB)
#      │
#      ▼
# Add summary memory
#      │
#      ▼
# Add last few messages
#      │
#      ▼
# Send to LLM
# Sending the entire chat history every time is inefficient and will quickly hit the context/token limit of the model.
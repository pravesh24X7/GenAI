import json
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    SystemMessage
)
from langchain_groq import ChatGroq
from currency_converter_tool import CurrencyToolkit
from dotenv import load_dotenv

load_dotenv()

# Load tools
tools = CurrencyToolkit().get_tools()

# LLM
llm_model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.35
)

llm_model_with_tool = llm_model.bind_tools(tools)

# Chat history
chat_history = [
    SystemMessage("You are a helpful assistant for currency conversion.")
]

# -------- Helper --------
def safe_float(x):
    try:
        return float(x)
    except:
        return None


def get_tool(tool_name):
    return [tool for tool in tools if tool.name == tool_name][0]


# -------- Chat Loop --------
while True:
    query = input("Human : ").strip()

    if query.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=query))

    max_iters = 5
    iters = 0

    response = llm_model_with_tool.invoke(chat_history)

    while response.tool_calls and iters < max_iters:
        for tool_call in response.tool_calls:
            fxn = [tool.func for tool in tools if tool.name == tool_call["name"]][0]

            result = fxn(**tool_call["args"])

            chat_history.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

        response = llm_model_with_tool.invoke(chat_history)
        iters += 1

    print("AI :", response.content)
    chat_history.append(response)

print("Program Terminated.")
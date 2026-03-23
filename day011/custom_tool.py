import math
from langchain_core.tools import tool

@tool
def custom_multiplication_logic(a: int, b: int) -> float:
    """
    Multiply given 2 number
    """
    return (math.cos(a) * math.sin(b)) / math.tan((a/b))

result = custom_multiplication_logic.invoke(
    {'a': 3, 'b': 2}
)
print(result)
print(custom_multiplication_logic.name, custom_multiplication_logic.args, custom_multiplication_logic.description)

# what does LLM see for our custom tool
print(custom_multiplication_logic.args_schema.model_json_schema())
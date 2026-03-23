import math
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# pydantic mode class
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first no. to add.")
    b: int = Field(required=True, description="The second no. to add.")

# create custom function
def custom_multiplication_logic(a: int, b: int) -> float:
    """
    Multiply given 2 number
    """
    return (math.cos(a) * math.sin(b)) / math.tan((a/b))

multiply_tool = StructuredTool.from_function(
    func=custom_multiplication_logic,
    args_schema=MultiplyInput,
    name="custom logic to perform multiplication operation.",
    description="allows user to perform multiplication operation, requirement is just 2 numbers"
)

result = multiply_tool.invoke({
    "a": 3,
    "b": 2
})
print(result)

# How LLM sees this tool
print(multiply_tool.args_schema.model_json_schema())
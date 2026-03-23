import math
from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class MutliplyInput(BaseModel):
    a: int = Field(required=True, description="First no. to multiply.")
    b: int = Field(required=True, description="Second no. to multiply.")


class MultiplyTool(BaseTool):
    name: str = "custom_multiply_tool"
    description: str = "Used to perform multiplication using custom logic."

    args_schema: Type[BaseModel] = MutliplyInput

    def _run(self, a: int, b: int) -> float:
        return (math.cos(a) * math.sin(b)) / math.tan((a/b))

mt = MultiplyTool()
results = mt.invoke({'a': 3, 'b': 2})

print(results)
print(mt.args_schema.model_json_schema())
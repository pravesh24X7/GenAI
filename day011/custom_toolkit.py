import math
from langchain_core.tools import tool

# for creating toolkit first we've to create more than one tools.
@tool
def custom_add(a: int, b: int) -> float:
    """Performs addition using custom logic."""
    return math.sin(a) + math.tan(b)

@tool
def custom_div(a: int, b: int) -> float:
    """Performs division using custom logic."""
    return math.tan(a) / math.cos(b)

# create toolkit class
class MathToolkit:
    def get_tools(self):
        return [custom_add, custom_div]

toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(f"Name: {tool.name}\nDescription: {tool.description}")
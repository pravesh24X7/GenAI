import math
from langchain_core.tools import tool

# creating tool
@tool
def custom_addition_fxn(a: int, b: int) -> float:
    """
    Custom addition fxn: responsible for performing special addition operations using trignometric operations.
    a (int)     :       first number of type integer.
    b(int)      :       second number of type integer.

    returns:
    result (float)  :   Special addition operation result
    """
    result = math.sin(a) + math.tan(b)
    return result
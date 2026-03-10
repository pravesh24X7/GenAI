from typing import TypedDict
from typing import List, Annotated

class QA(TypedDict):
    answer: Annotated[str, "a short or one word answer."]

# usage
# sports: SportsGenre = {
#     "answer": "Cricket"
# }
from langgraph.graph import StateGraph, START, END      # START and END are dummy nodes which tells langgraph where graph begins and where it ends.
from typing import TypedDict



# define state
class BMIState(TypedDict):
    weight: float
    height: float
    bmi: float
    category: str


# functions
def calculate_bmi(state: BMIState) -> BMIState:
    
    current_weight = state['weight']
    current_height = state['height']
    
    bmi = current_weight / (current_height ** 2)
    state['bmi'] = round(bmi, 2)
    return state


def label_bmi(state: BMIState) -> BMIState:

    bmi = state['bmi']

    if bmi < 18.5:
        state["category"] = "under-weight"
    elif 18.5 <= bmi < 25:
        state['category'] = "normal"
    elif 25 <= bmi < 30:
        state['category'] = "over-weight"
    else:
        state['category'] = "obese"
    return state



# define graph
graph = StateGraph(BMIState)

# add nodes
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('label_bmi', label_bmi)

# add edges
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', 'label_bmi')
graph.add_edge('label_bmi', END)

# compile graph
workflow = graph.compile()

# begin execution
# height, weight = tuple(
#     map(float, input(" Enter your height( in meters ) and weight ( in kilograms ) :").strip().split())
# )

initial_state = {
    "weight": 75,
    "height": 1.77800,
}
final_state = workflow.invoke(initial_state)

print(final_state)

# display the graph
img_file = workflow.get_graph().draw_mermaid_png()

with open("./img001.png", "wb") as image:
    image.write(img_file)
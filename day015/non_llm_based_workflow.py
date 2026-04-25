from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# define state
class PlayerState(TypedDict):
    runs: int
    balls: int
    _4s: int
    _6s: int
    strike_rate: float
    boundary_per_ball: int
    runs_boundary_percentage: float


# functions

def calculate_strike_rate(state: PlayerState) -> dict:
    strike_rate = state["runs"] / state["balls"]
    return {"strike_rate": (strike_rate*100)}


def calculate_boundary_per_ball(state: PlayerState) -> dict:
    bpb = state["balls"] / (state["_4s"] + state["_6s"])
    return {"boundary_per_ball": (bpb*100)}


def calculate_runs_boundary_percentage(state: PlayerState) -> dict:
    boundary_runs = (state["_4s"]*4) + (state["_6s"]*6)
    runs_boundary_percentage = ((boundary_runs)/state["runs"])*100

    return {"runs_boundary_percentage": runs_boundary_percentage}   # this is called partial updates


# create execution graph
graph = StateGraph(PlayerState)

# add nodes
graph.add_node("strike_rate", calculate_strike_rate)
graph.add_node("boundary_per_ball", calculate_boundary_per_ball)
graph.add_node("runs_boundary_percentage", calculate_runs_boundary_percentage)

# add edges
graph.add_edge(START, "strike_rate")
graph.add_edge(START, "boundary_per_ball")
graph.add_edge(START, "runs_boundary_percentage")
graph.add_edge("strike_rate", END)
graph.add_edge("boundary_per_ball", END)
graph.add_edge("runs_boundary_percentage", END)

# compile graph
workflow = graph.compile()

# invoke
initial_state = {
    "runs": 171,
    "balls": 93,
    "_4s": 23,
    "_6s": 9,
}

final_state = workflow.invoke(initial_state)
print(final_state)

# print the graph
img = workflow.get_graph().draw_mermaid_png()

with open("./img001.png", "wb") as f:
    f.write(img)
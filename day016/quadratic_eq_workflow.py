"""
Quadratic Equations:
    while solving an equation of form:
            aX**2 + bX + c = 0
    what solution will exists depends upon the discriminant (D = [b**2 - 4ac])

            If D > 0:
                2 distinct real roots
                these are 
                    1. -b + (D) ** 0.5 / 2a
                    2. -b - (D) ** 0.5 / 2a
            IF D == 0:
                1 repeated root
                that is:
                    -b / 2a
            Else:
                no real roots
"""

import math
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal


class QuadEq(TypedDict):
    coef_a: float
    coef_b: float
    coef_c: float
    
    discriminant: float
    equation: str
    result: str



def show_equation(state: QuadEq) -> QuadEq:

    equation = (f"{state['coef_a']}X**2 {'+' if state['coef_b'] > 0 else ''}{state['coef_b']}X {'+' if state['coef_c'] > 0 else ''}{state['coef_c']}")

    state["equation"] = equation
    return state


def calculate_discriminant(state: QuadEq) -> dict:

    discriminant = (state['coef_b'] ** 2) - (4 * state['coef_a'] * state['coef_c'])
    return {"discriminant": discriminant}


def no_real_root(state: QuadEq) -> dict:
    return {"result": "No Real Roots Exists for the given equation."}


def real_root(state: QuadEq) -> dict:
    
    root1 = (-state["coef_b"] + math.sqrt(state["discriminant"])) / (2 * state["coef_a"])
    root2 = (-state["coef_b"] - math.sqrt(state["discriminant"])) / (2 * state["coef_a"])

    return {"result": f"Two real roots are: {root1} and {root2}"}


def repeated_root(state: QuadEq) -> dict:
    return {"result": f"{(-state['coef_b']) / (2 * state['coef_a'])}"}


def check_condition(state: QuadEq) -> Literal["real_root", "no_real_root", "repeated_root"]:
    
    if state["discriminant"] > 0:
        return "real_root"
    elif state["discriminant"] == 0:
        return "repeated_root"
    else:
        return "no_real_root"
    

# defining graph
graph = StateGraph(QuadEq)

# add nodes
graph.add_node("show_equation", show_equation)
graph.add_node("calculate_discriminant", calculate_discriminant)
graph.add_node("no_real_root", no_real_root)
graph.add_node("real_root", real_root)
graph.add_node("repeated_root", repeated_root)

# add edges
graph.add_edge(START, "show_equation")
graph.add_edge("show_equation", "calculate_discriminant")
graph.add_conditional_edges("calculate_discriminant", check_condition)
graph.add_edge("real_root", END)
graph.add_edge("no_real_root", END)
graph.add_edge("repeated_root", END)


# compile graph
workflow = graph.compile()

# begin execution
initial_state = {
    "coef_a": 4,
    "coef_b": 2,
    "coef_c": 2
}

final_state = workflow.invoke(initial_state)
print(final_state)


# save the graph
img = workflow.get_graph().draw_mermaid_png()
with open("./img001.png", mode="wb") as f:
    f.write(img)
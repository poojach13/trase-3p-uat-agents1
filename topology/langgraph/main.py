"""UAT topology fixture: a LangGraph StateGraph with a conditional branch, compiled and
exported at module level as /guides/frameworks/overview/ requires for graph extraction.

No model call anywhere; run() returns a fixed value. The platform inspects ``graph`` at
build time through the runnable-graph adapter (get_graph()).
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    text: str
    topic: str


def classify(state: State) -> dict:
    return {"topic": "billing" if "invoice" in state["text"] else "returns"}


def billing(state: State) -> dict:
    return {"text": "billing: " + state["text"]}


def returns(state: State) -> dict:
    return {"text": "returns: " + state["text"]}


def reply(state: State) -> dict:
    return {"text": state["text"].upper()}


def route(state: State) -> str:
    return state["topic"]


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("billing", billing)
builder.add_node("returns", returns)
builder.add_node("reply", reply)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route, {"billing": "billing", "returns": "returns"})
builder.add_edge("billing", "reply")
builder.add_edge("returns", "reply")
builder.add_edge("reply", END)
graph = builder.compile()


def run() -> str:
    """Never invokes a model; the platform inspects ``graph`` at build time."""
    return "langgraph-topology:ok"

"""UAT topology fixture. Copied from the platform's own inspector fixture
(agent-build-service/.../topology/strands_fixture/agent.py) so the expected topology is engineering's
`expected.json` beside this file. Built at import and never run against a model:
run() only returns a fixed value."""

from strands import Agent, tool
from strands.multiagent import GraphBuilder, Swarm


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return order_id


@tool
def handoff_to_user(message: str) -> str:
    """Hand the turn to a person (the strands_tools HITL shape)."""
    return message


def needs_review(state) -> bool:
    """Route to review when the draft needs it."""
    del state
    return True


triage = Agent(name="triage", system_prompt="Route the customer.", tools=[lookup_order])
billing = Agent(name="billing", system_prompt="Handle billing.", tools=[handoff_to_user])
returns = Agent(name="returns", system_prompt="Handle returns.")
support = Swarm([billing, returns], entry_point=billing)
writer = Agent(name="writer", system_prompt="Write the reply.")

_builder = GraphBuilder()
_builder.add_node(triage, "triage")
_builder.add_node(support, "support")
_builder.add_node(writer, "writer")
_builder.add_edge("triage", "support", condition=needs_review)
_builder.add_edge("support", "writer")
_builder.set_entry_point("triage")
root_agent = _builder.build()


def run() -> str:
    """Never invokes a model; the platform inspects the module-level object at build time."""
    return "strands-topology:ok"

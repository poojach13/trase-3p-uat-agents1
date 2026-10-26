"""UAT topology fixture. Copied from the platform's own inspector fixture
(agent-build-service/.../topology/adk_fixture/agent.py) so the expected topology is engineering's
`expected.json` beside this file. Built at import and never run against a model:
run() only returns a fixed value."""

from google.adk.agents import Agent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.tools import AgentTool, FunctionTool, LongRunningFunctionTool
from google.adk.tools.base_toolset import BaseToolset


def lookup_order(order_id: str) -> dict:
    """Look up an order."""
    return {"order_id": order_id}


def request_refund(order_id: str) -> dict:
    """Ask a human to approve a refund."""
    return {"order_id": order_id, "status": "pending"}


class StubToolset(BaseToolset):
    """A toolset the inspector must list without resolving."""

    async def get_tools(self, readonly_context=None):
        """Fail loudly: the inspector lists toolsets, it never resolves them."""
        del readonly_context
        raise AssertionError("toolsets must never be resolved during inspection")

    async def close(self):
        """Nothing to release."""
        return


summarizer = Agent(name="summarizer", model="gemini-2.5-flash", instruction="Summarize.")
billing = Agent(name="billing", model="gemini-2.5-flash", instruction="Handle billing.")
returns = Agent(
    name="returns",
    model="gemini-2.5-flash",
    instruction="Handle returns.",
    disallow_transfer_to_parent=True,
)
triage = Agent(
    name="triage",
    model="gemini-2.5-flash",
    instruction="Route the customer.",
    tools=[
        lookup_order,
        LongRunningFunctionTool(request_refund),
        FunctionTool(lookup_order),
        AgentTool(agent=summarizer),
        StubToolset(),
    ],
    sub_agents=[billing, returns],
)
critic = Agent(name="critic", model="gemini-2.5-flash", instruction="Critique.")
fixer = Agent(name="fixer", model="gemini-2.5-flash", instruction="Fix.")
polish = LoopAgent(name="polish", sub_agents=[critic, fixer], max_iterations=3)
fact = Agent(name="fact", model="gemini-2.5-flash")
style = Agent(name="style", model="gemini-2.5-flash")
checks = ParallelAgent(name="checks", sub_agents=[fact, style])
root_agent = SequentialAgent(name="pipeline", sub_agents=[triage, polish, checks])


def run() -> str:
    """Never invokes a model; the platform inspects the module-level object at build time."""
    return "adk-topology:ok"

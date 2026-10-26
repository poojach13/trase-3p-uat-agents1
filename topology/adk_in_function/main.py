"""UAT topology fixture (negative): the ADK agent is only built inside run(), so there is no
module-level ``root_agent`` to inspect. The platform must report the topology as unsupported
with a note telling the author to expose ``root_agent`` (PLT-5598, PLT-5013), not draw a
made-up graph. run() never calls a model.
"""

from google.adk.agents import Agent


def _make_agent() -> Agent:
    return Agent(name="hidden", model="gemini-2.5-flash", instruction="Hidden.")


def run() -> str:
    agent = _make_agent()
    return f"adk_in_function-topology:ok:{agent.name}"

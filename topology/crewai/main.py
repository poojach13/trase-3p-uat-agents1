"""UAT topology fixture. Copied from the platform's own inspector fixture
(agent-build-service/.../topology/crewai_fixture/agent.py) so the expected topology is engineering's
`expected.json` beside this file. Built at import and never run against a model:
run() only returns a fixed value."""

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool


class LookupOrder(BaseTool):
    """A stock crewAI tool; never run during inspection."""

    name: str = "lookup_order"
    description: str = "Look up an order."

    def _run(self, order_id: str) -> str:  # noqa: PLR6301 — crewAI's BaseTool contract
        return order_id


researcher = Agent(
    role="Researcher",
    goal="Find the facts behind the customer's question.",
    backstory="Thorough and curious.",
    tools=[LookupOrder()],
    llm="gpt-4o",
)
writer = Agent(
    role="Writer",
    goal="Write a short, accurate reply.",
    backstory="Terse.",
    llm="gpt-4o",
    allow_delegation=True,
)
research = Task(description="Research the question.", expected_output="facts", agent=researcher)
outline = Task(description="Outline the answer.", expected_output="outline", agent=researcher)
draft = Task(
    description="Write the reply.",
    expected_output="reply",
    agent=writer,
    context=[research, outline],
    human_input=True,
)
crew = Crew(
    agents=[researcher, writer], tasks=[research, outline, draft], process=Process.sequential
)


def run() -> str:
    """Never kicks the crew off; the platform inspects ``crew`` at build time."""
    return "crewai-topology:ok"

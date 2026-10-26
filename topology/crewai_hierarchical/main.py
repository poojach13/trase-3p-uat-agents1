"""UAT topology fixture: a hierarchical crewAI crew (PLT-5652: Process.hierarchical adds a
manager node with delegate edges to and from every agent). Constructed at import, never
kicked off; run() returns a fixed value.
"""

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool


class LookupOrder(BaseTool):
    """A stock crewAI tool; never run."""

    name: str = "lookup_order"
    description: str = "Look up an order."

    def _run(self, order_id: str) -> str:
        return order_id


researcher = Agent(role="Researcher", goal="Find the facts.", backstory="Thorough.",
                   tools=[LookupOrder()], llm="gpt-4o")
writer = Agent(role="Writer", goal="Write the reply.", backstory="Terse.", llm="gpt-4o")
research = Task(description="Research the question.", expected_output="facts")
draft = Task(description="Write the reply.", expected_output="reply")
crew = Crew(agents=[researcher, writer], tasks=[research, draft],
            process=Process.hierarchical, manager_llm="gpt-4o")


def run() -> str:
    """Never kicks the crew off; the platform inspects ``crew`` at build time."""
    return "crewai_hierarchical-topology:ok"

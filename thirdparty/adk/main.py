"""Google ADK 3P agent for the UAT 3P suite: a custom BaseAgent run by an ADK Runner (no model
call). The run input is passed as the user message; the agent replies with the JSON result."""

import asyncio
import json
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

import uat_common

APP = "uat_3p"


class UatAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        data = json.loads(ctx.user_content.parts[0].text)
        result = uat_common.dispatch(data, "adk", lambda t: t.upper())
        yield Event(author=self.name, invocation_id=ctx.invocation_id,
                    content=types.Content(role="model", parts=[types.Part(text=json.dumps(result))]))


root_agent = UatAgent(name="uat_agent")


async def _main(data: dict) -> dict:
    runner = InMemoryRunner(agent=root_agent, app_name=APP)
    session = await runner.session_service.create_session(app_name=APP, user_id="uat")
    message = types.Content(role="user", parts=[types.Part(text=json.dumps(data))])
    answer = "{}"
    async for event in runner.run_async(user_id="uat", session_id=session.id, new_message=message):
        if event.content and event.content.parts and event.content.parts[0].text is not None:
            answer = event.content.parts[0].text
    return json.loads(answer)


def run():
    return asyncio.run(_main(uat_common.read()))

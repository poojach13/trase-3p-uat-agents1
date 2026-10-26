"""LangChain 3P agent for the UAT 3P suite. Every mode runs through a LangChain tool inside a
runnable chain, so framework: langchain auto-instrumentation records TOOL_* events."""

import json

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

import uat_common


@tool
def shout(text: str) -> str:
    """Upper-case the text."""
    return text.upper()


@tool
def uat_action(request: str) -> str:
    """Run one UAT mode (calls, MCP, chat, logs) described by a JSON request."""
    return json.dumps(uat_common.dispatch(json.loads(request), "langchain",
                                          lambda t: shout.invoke({"text": t})))


chain = RunnableLambda(lambda data: json.loads(uat_action.invoke({"request": json.dumps(data)})))


def run():
    return chain.invoke(uat_common.read())

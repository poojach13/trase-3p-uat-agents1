"""Shared by the UAT 3P agents (LangChain and ADK). Standard library + trase_os_sdk.sandbox only.

One deployed agent covers every 3P story, chosen by the run input's "mode":
  {"mode": "shout", "text": T}                 -> {"shout": T.upper()}
  {"mode": "call", "calls": [{handle, path, method?, body?, auth?}]}
                                                -> one result per call (status, refused_by, body)
  {"mode": "mcp", "handle": H, "path": "/mcp"} -> MCP initialize + tools/list over the gateway
  {"mode": "chat", "handle": H, "base_path": P, "model": M}
                                                -> one OpenAI-compatible chat call ("pong")
  {"mode": "log", "marker": X}                 -> one log line per level, each carrying X
  {"mode": "fail", "marker": X}                -> logs an ERROR line, then raises
  {"mode": "whoami"}                           -> a fingerprint of this run's credential
  {"mode": "echo", ...anything}                -> the input exactly as read, its sha256, value types
  {"mode": "big", "kb": N}                     -> an N KB output (default 64), with its sha256
  {"mode": "mcp_call", "handle": H, "tool": T, "arguments": {...}}
                                                -> MCP initialize + tools/call T over the gateway

Calls never raise on HTTP errors, so tests can assert on refusal codes. The run credential is
never returned; only a short SHA-256 fingerprint and whether an upstream echoed it.
"""

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request

from trase_os_sdk.sandbox import NoInputError, connection_url, read_input, run_credential

log = logging.getLogger("uat_agent")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def read() -> dict:
    try:
        data = read_input()
    except NoInputError:
        data = {}
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            data = {"text": data}
    return data or {}


def _http(url: str, *, method: str = "GET", body=None, auth: bool = True, headers=None) -> dict:
    h = {"Accept": "application/json, text/event-stream", "User-Agent": "trase-uat"}
    h.update(headers or {})
    if auth:
        h["Authorization"] = f"Bearer {run_credential()}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            status, rh, raw = resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as err:
        status, rh, raw = err.code, err.headers, err.read()
    except urllib.error.URLError as err:
        return {"status": 0, "refused_by": None, "body": f"URLError: {err.reason}", "credential_echoed": False}
    text = raw.decode("utf-8", "replace")
    return {"status": status, "refused_by": rh.get("x-trase-refused-by"),
            "session": rh.get("mcp-session-id"), "body": text[:2000], "text": text,
            "credential_echoed": run_credential() in text}


def call(spec: dict) -> dict:
    url = f"{connection_url(spec['handle'])}{spec.get('path', '/get')}"
    out = _http(url, method=spec.get("method", "GET"), body=spec.get("body"), auth=spec.get("auth", True))
    log.info("uat call %s %s -> %s", spec["handle"], spec.get("path", "/get"), out["status"])
    return {"handle": spec["handle"], **out}


def _jsonrpc_messages(out: dict) -> tuple[list[dict], list[str]]:
    """Every JSON-RPC message in a response, parsed from the FULL body (not the 2,000-char excerpt):
    the SSE `data:` lines, or the whole body when it isn't SSE. Parse failures are returned, never
    swallowed (PLT-6090: a truncated tools/list line used to read as an empty tool list)."""
    text = out.get("text", out.get("body", ""))
    sse = [line[5:].strip() for line in text.split("\n") if line.startswith("data:")]
    messages, errors = [], []
    for chunk in sse or [text.strip()]:
        if not chunk:
            continue
        try:
            msg = json.loads(chunk)
        except ValueError as err:
            errors.append(f"{type(err).__name__}: {err}"[:200])
            continue
        if isinstance(msg, dict):
            messages.append(msg)
    return messages, errors


def mcp(handle: str, path: str = "/mcp") -> dict:
    url = f"{connection_url(handle)}{path}"
    init = _http(url, method="POST", body={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                   "clientInfo": {"name": "trase-uat", "version": "1"}}})
    extra = {"Mcp-Session-Id": init["session"]} if init.get("session") else {}
    _http(url, method="POST", body={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=extra)
    tools = _http(url, method="POST", body={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, headers=extra)
    messages, parse_errors = _jsonrpc_messages(tools)
    names = [t.get("name") for m in messages for t in ((m.get("result") or {}).get("tools") or [])
             if isinstance(t, dict)]
    if parse_errors:
        log.error("uat mcp %s tools/list: %d unparseable line(s): %s", handle, len(parse_errors), parse_errors[0])
    return {"handle": handle, "initialize_status": init["status"], "tools_status": tools["status"],
            "refused_by": init["refused_by"] or tools["refused_by"], "tools": names,
            "tools_parse_errors": parse_errors, "tools_list_bytes": len(tools.get("text", "")),
            "body": init["body"][:500]}


def chat(handle: str, base_path: str, model: str) -> dict:
    url = f"{connection_url(handle)}{base_path}/chat/completions"
    out = _http(url, method="POST", body={
        "model": model, "max_tokens": 10,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}]})
    reply = ""
    try:
        reply = json.loads(out.get("text", out["body"]))["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        pass
    if out["status"] >= 400:
        log.error("uat chat via %s failed: HTTP %s %s", handle, out["status"], out["body"][:300])
    return {"handle": handle, "model": model, "reply": reply, **out}


def emit_logs(marker: str) -> dict:
    log.debug("uat-log DEBUG %s", marker)
    log.info("uat-log INFO %s", marker)
    log.warning("uat-log WARNING %s", marker)
    log.error("uat-log ERROR %s", marker)
    print(f"uat-log STDOUT {marker}", flush=True)
    return {"marker": marker, "levels": ["DEBUG", "INFO", "WARNING", "ERROR", "STDOUT"]}


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def echo(data: dict) -> dict:
    """The input exactly as read_input() returned it, so tests can prove nothing was changed."""
    return {"received": data, "sha256": _sha(data), "keys": sorted(data),
            "types": {k: type(v).__name__ for k, v in data.items()},
            "gateway": bool(os.environ.get("TRASE_EGRESS_GATEWAY_URL"))}


def big(kb: int) -> dict:
    blob = ("0123456789abcdef" * 64 * max(1, int(kb)))[: max(1, int(kb)) * 1024]
    return {"size": len(blob), "sha256": hashlib.sha256(blob.encode()).hexdigest(), "blob": blob}


def mcp_call(handle: str, tool_name: str, arguments: dict, path: str = "/mcp") -> dict:
    url = f"{connection_url(handle)}{path}"
    init = _http(url, method="POST", body={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                   "clientInfo": {"name": "trase-uat", "version": "1"}}})
    extra = {"Mcp-Session-Id": init["session"]} if init.get("session") else {}
    _http(url, method="POST", body={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=extra)
    res = _http(url, method="POST", headers=extra, body={
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments or {}}})
    result, error = None, None
    messages, parse_errors = _jsonrpc_messages(res)
    for msg in messages:
        result, error = msg.get("result", result), msg.get("error", error)
    if parse_errors:
        log.error("uat mcp_call %s %s: %d unparseable line(s): %s", handle, tool_name, len(parse_errors), parse_errors[0])
    log.info("uat mcp_call %s %s -> %s", handle, tool_name, res["status"])
    return {"handle": handle, "tool": tool_name, "initialize_status": init["status"],
            "call_status": res["status"], "refused_by": init["refused_by"] or res["refused_by"],
            "result": result, "error": error, "parse_errors": parse_errors, "body": res["body"][:800]}


def direct(url: str) -> dict:
    """Call an upstream directly, around the gateway (MCP initialize). The sandbox should refuse it."""
    try:
        out = _http(url, method="POST", auth=False, body={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                       "clientInfo": {"name": "trase-uat", "version": "1"}}})
    except Exception as err:  # noqa: BLE001 - a reset or timeout IS the expected outcome
        out = {"status": 0, "refused_by": None, "body": f"{type(err).__name__}: {err}"[:300]}
    log.info("uat direct %s -> %s", url, out["status"])
    return {"url": url, "status": out["status"], "body": str(out.get("body", ""))[:300]}


def whoami() -> dict:
    cred = run_credential()
    return {"credential_fingerprint": hashlib.sha256(cred.encode()).hexdigest()[:16],
            "gateway": bool(os.environ.get("TRASE_EGRESS_GATEWAY_URL"))}


def dispatch(data: dict, framework: str, shout) -> dict:
    """Run one mode. `shout` is the framework-native path for the plain run."""
    mode = data.get("mode", "shout")
    if mode == "call":
        result = {"calls": [call(c) for c in data.get("calls", [])]}
    elif mode == "mcp":
        result = mcp(data["handle"], data.get("path", "/mcp"))
    elif mode == "chat":
        result = chat(data["handle"], data.get("base_path", "/v1"), data["model"])
    elif mode == "log":
        result = emit_logs(data.get("marker", "none"))
    elif mode == "fail":
        log.error("uat-fail about to raise %s", data.get("marker", "none"))
        raise RuntimeError(f"uat-deliberate-failure {data.get('marker', 'none')}")
    elif mode == "whoami":
        result = whoami()
    elif mode == "echo":
        result = echo(data)
    elif mode == "big":
        result = big(data.get("kb", 64))
    elif mode == "direct":
        result = direct(data["url"])
    elif mode == "mcp_call":
        result = mcp_call(data["handle"], data["tool"], data.get("arguments") or {}, data.get("path", "/mcp"))
    else:
        result = {"shout": shout(data.get("text") or data.get("user_message", ""))}
    return {"mode": mode, "framework": framework, **result}

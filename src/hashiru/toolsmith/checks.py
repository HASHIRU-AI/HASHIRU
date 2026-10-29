"""Pure checks on what agents and tool processes hand the hub. Nothing here runs agent-written code."""
import ast
import hashlib
import json
import keyword
import re
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement

__all__ = ["MAX_RESULT_CHARS", "NAME", "check_entry", "check_published", "check_request", "digest", "to_result"]

NAME = re.compile(r"[a-z][a-z0-9_]{0,31}")
MAX_CODE_BYTES = 100_000
MAX_DEPENDENCIES = 10
MAX_DESCRIPTION_CHARS = 1_000   # every manager in every session reads it on every turn
MAX_SCHEMA_CHARS = 10_000
MAX_PARAMETERS = 32
MAX_RESULT_CHARS = 20_000       # what one call may put in an agent's context


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_request(name: Any, code: Any, dependencies: Any, taken: set[str]) -> str | None:
    """Why a create_tool request (or a saved tool) is refused, or None. The code is parsed, never run."""
    if not isinstance(name, str) or not NAME.fullmatch(name) or keyword.iskeyword(name):
        return f"name {name!r} must start with a-z and use only a-z, 0-9 and _ (at most 32)"
    if name in taken:
        return f"a tool named {name!r} already exists"
    if "__" in name:
        return "names may not contain '__', which marks tools of configured MCP servers"
    if not isinstance(code, str):
        return "code must be text"
    try:
        size = len(code.encode())
    except UnicodeEncodeError:
        return "the code is not valid Unicode text"
    if size > MAX_CODE_BYTES:
        return f"code is larger than {MAX_CODE_BYTES} bytes"
    try:
        tree = ast.parse(code, filename=f"{name}.py")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        return f"the code does not parse: {type(exc).__name__}: {exc}"
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}
    if name not in defined:
        return f"the code must define a top-level function named {name}"
    if not isinstance(dependencies, list) or len(dependencies) > MAX_DEPENDENCIES:
        return f"dependencies must be a list of at most {MAX_DEPENDENCIES} requirements"
    for dep in dependencies:
        if not isinstance(dep, str):
            return "each dependency must be a string such as 'requests==2.32.3'"
        try:
            requirement = Requirement(dep)
        except InvalidRequirement as exc:
            return f"{dep!r} is not a valid requirement: {exc}"
        if requirement.url:
            return f"{dep!r}: dependencies must come from PyPI, not a URL"
    return None


def check_entry(entry: Any) -> str | None:
    """Why a manifest entry cannot be loaded, or None. The manifest is a file tools could have edited."""
    if not isinstance(entry, dict):
        return "its manifest entry is not an object"
    if not isinstance(entry.get("sha256"), str):
        return "its manifest entry has no sha256"
    if not isinstance(entry.get("dependencies", []), list):
        return "its manifest dependencies are not a list"
    return None


def check_published(description: str, schema: dict[str, Any]) -> str | None:
    """Why a tool's self-description is refused: it reaches every manager's prompt, so it must stay small."""
    properties, required = schema.get("properties", {}), schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list) \
            or not all(isinstance(r, str) for r in required):
        return "the parameter schema is malformed"
    if len(description) > MAX_DESCRIPTION_CHARS:
        return f"the docstring has {len(description)} characters; keep it under {MAX_DESCRIPTION_CHARS}"
    if len(schema.get("properties", {})) > MAX_PARAMETERS:
        return f"the function has more than {MAX_PARAMETERS} parameters"
    if len(json.dumps(schema)) > MAX_SCHEMA_CHARS:
        return f"the parameter schema is larger than {MAX_SCHEMA_CHARS} characters"
    return None


def to_result(result: Any) -> dict[str, Any]:
    """A tool process's CallToolResult -> the dict the agent sees, cut to MAX_RESULT_CHARS when serialized."""
    blocks = getattr(result, "content", None) or []
    texts = [b.text for b in blocks if getattr(b, "type", None) == "text"]
    others = sorted({str(getattr(b, "type", "unknown")) for b in blocks if getattr(b, "type", None) != "text"})
    structured = getattr(result, "structured_content", None)
    out: dict[str, Any]
    if getattr(result, "is_error", False):
        out = {"error": "\n".join(texts) or "the tool failed"}
    elif isinstance(structured, dict):
        out = dict(structured)
    elif len(texts) == 1:
        value = _json_or_text(texts[0])  # a bare `-> dict` arrives as JSON text
        out = value if isinstance(value, dict) else {"result": value}
    else:
        out = {"result": [_json_or_text(t) for t in texts]}  # e.g. a list returned without annotation
    if others:
        out["non_text_content"] = others
    return _capped(out)


def _json_or_text(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def _capped(out: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(out, default=str)
    if len(text) <= MAX_RESULT_CHARS:
        return out
    key = "error" if "error" in out else "result"
    return {key: text[:MAX_RESULT_CHARS], "truncated": f"{len(text)} characters cut to {MAX_RESULT_CHARS}"}

"""Build a hub-side proxy function whose signature mirrors a child tool's JSON schema.

MCPServer derives a tool's schema from a function signature, and the hub must never import agent-written
code, so the signature is synthesized from the schema the child reported. The child validates again with
its exact schema; the proxy only has to carry the names, basic types, descriptions and defaults.
"""
import keyword
import re
from collections.abc import Awaitable, Callable
from inspect import Parameter, Signature
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

__all__ = ["make_proxy", "py_type", "register_proxy"]

JSON_TYPES: dict[str, Any] = {"string": str, "integer": int, "number": float, "boolean": bool,
                              "array": list, "object": dict}

PARAM = re.compile(r"[A-Za-z][A-Za-z0-9_]*")  # pydantic rejects a leading underscore and model_ names

Forward = Callable[[Context, str, dict[str, Any]], Awaitable[dict[str, Any]]]


def py_type(spec: dict[str, Any]) -> Any:
    """JSON schema -> a Python annotation good enough for argument validation."""
    kind = spec.get("type")
    if isinstance(kind, str):
        return JSON_TYPES.get(kind, Any)
    options = spec.get("anyOf") or ([{"type": k} for k in kind] if isinstance(kind, list) else [])
    options = [o if isinstance(o, dict) else {} for o in options]  # `true` is a valid (match-all) subschema
    nullable = any(o.get("type") == "null" for o in options)
    kinds = [py_type(o) for o in options if o.get("type") != "null"]
    base = kinds[0] if len(kinds) == 1 else Any
    return base | None if nullable and base is not Any else base


def make_proxy(name: str, schema: dict[str, Any], description: str, forward: Forward) -> Callable[..., Any]:
    """An async function with one keyword parameter per schema property (plus the MCP ctx)."""
    properties, required = schema.get("properties", {}), schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ValueError("the parameter schema is malformed")
    params = [Parameter("ctx", Parameter.KEYWORD_ONLY, annotation=Context)]
    droppable: set[str] = set()  # optional with no default (or None): omit, so the child applies its own
    for prop, raw_spec in properties.items():
        spec = raw_spec if isinstance(raw_spec, dict) else {}  # `true` is a valid (match-all) subschema
        if not PARAM.fullmatch(prop) or keyword.iskeyword(prop) or prop == "ctx" or prop.startswith("model_"):
            raise ValueError(f"parameter name {prop!r} cannot be used")
        # Built at runtime from the child's schema, which a static checker cannot follow.
        annotation = Annotated[py_type(spec), Field(description=spec.get("description"))]  # type: ignore[valid-type]
        if prop in required:
            params.append(Parameter(prop, Parameter.KEYWORD_ONLY, annotation=annotation))
        else:
            if spec.get("default") is None:
                droppable.add(prop)
            params.append(Parameter(prop, Parameter.KEYWORD_ONLY, annotation=annotation,
                                    default=spec.get("default")))

    async def proxy(ctx: Context, **kwargs: Any) -> dict[str, Any]:
        # None for a parameter whose default is None means "omitted"; for one with another default (2.0, say),
        # None was passed on purpose and must reach the child.
        args = {k: v for k, v in kwargs.items() if not (v is None and k in droppable)}
        return await forward(ctx, name, args)

    proxy.__signature__ = Signature(params, return_annotation=dict[str, Any])  # type: ignore[attr-defined]
    proxy.__annotations__ = {p.name: p.annotation for p in params} | {"return": dict[str, Any]}
    proxy.__name__ = name
    proxy.__doc__ = description
    return proxy


def register_proxy(mcp: MCPServer, name: str, description: str, schema: dict[str, Any], forward: Forward,
                   annotations: ToolAnnotations | None = None) -> None:
    """Offer a remote tool on `mcp` under `name`, publishing its exact schema (enums, nested types).

    Raises ValueError if a parameter name cannot be a Python keyword argument.
    """
    proxy = make_proxy(name, schema, description, forward)
    try:
        mcp.add_tool(proxy, name=name, description=description, annotations=annotations)
    except Exception as exc:  # pydantic refuses some signatures; report them like any unusable schema
        raise ValueError(f"cannot offer {name}: {exc}") from exc
    registered = mcp._tool_manager.get_tool(name)  # mcp==2.2.0 is pinned; there is no public setter
    if registered is not None:
        registered.parameters = schema

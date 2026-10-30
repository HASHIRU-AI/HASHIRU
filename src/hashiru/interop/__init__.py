"""What HASHIRU offers the outside world, all on the hub's one port:

- /                              the chat page
- /v1                            an OpenAI-compatible Chat Completions API (model "hashiru")
- /.well-known/agent-card.json   the CEO as an A2A agent, served at /a2a (its card is also at
                                 /a2a/.well-known/agent-card.json, so other hubs can hire it by that URL)
- /public/mcp                    HASHIRU as an MCP server (tools `ask` and `end_session`)

Every one of them talks to the hub's Conversations. The auth gate guards them with the api_key setting. Each can
mark a conversation private (PRIVATE_HEADER, PRIVATE_KEY, the MCP ask argument): local models only from then on.
"""
from typing import TYPE_CHECKING

from hashiru.interop.a2a_public import A2A_PATH, PRIVATE_KEY, a2a_routes, public_card
from hashiru.interop.conversations import Busy, Conversation, Conversations
from hashiru.interop.mcp_public import MCP_PATH, PublicMCP
from hashiru.interop.openai_api import MODEL, PRIVATE_HEADER, SESSION_HEADER, OpenAIRoutes
from hashiru.interop.page import page_routes

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["A2A_PATH", "MCP_PATH", "MODEL", "PRIVATE_HEADER", "PRIVATE_KEY", "SESSION_HEADER", "Busy", "Conversation",
           "Conversations", "PublicMCP", "mount", "public_card"]


def mount(hub: "Hub") -> PublicMCP:
    """Add the public routes to the hub's app. Returns the MCP server, whose sessions the hub runs while serving."""
    public_mcp = PublicMCP(hub)
    hub.app.router.routes.extend([*page_routes(hub), *OpenAIRoutes(hub).routes(), *a2a_routes(hub),
                                  *public_mcp.routes])
    return public_mcp

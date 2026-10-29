"""Fetching from the web for agents: public pages only, every redirect checked, bodies capped."""
import asyncio
import ipaddress
import re
import socket
import xml.etree.ElementTree as ET
from typing import Any

import httpx

__all__ = ["fetch_page", "non_public", "search_arxiv"]

MAX_REDIRECTS = 5
MAX_PAGE_BYTES = 2_000_000
MAX_TEXT_CHARS = 20_000
ARXIV = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


async def non_public(url: str) -> str | None:
    """Why `url` must not be fetched, or None. Agents read untrusted pages that can name any address, so
    loopback (Ollama, the hub), private networks and link-local (cloud metadata) are refused.

    The name is resolved here and again by the HTTP client, so DNS rebinding between the two is not covered.
    """
    parsed = httpx.URL(url)
    if parsed.scheme not in ("http", "https") or not parsed.host:
        return f"refusing {url}: only http(s) URLs with a host"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parsed.host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        return f"could not resolve {parsed.host}: {exc}"
    for info in infos:
        if not ipaddress.ip_address(str(info[4][0]).split("%")[0]).is_global:
            return f"refusing {url}: {parsed.host} is not a public address"
    return None


async def read_capped(page: httpx.Response, limit: int) -> bytes:
    """At most `limit` bytes of a streamed body."""
    chunks, size = [], 0
    async for chunk in page.aiter_bytes():
        chunks.append(chunk)
        size += len(chunk)
        if size >= limit:
            break
    return b"".join(chunks)[:limit]


async def fetch(http: httpx.AsyncClient, url: str, params: dict[str, Any] | None = None) -> tuple[int, str]:
    """GET a public URL, following redirects by hand so every hop is checked. Raises ValueError if refused."""
    for _ in range(MAX_REDIRECTS + 1):
        if refused := await non_public(url):
            raise ValueError(refused)
        async with http.stream("GET", url, params=params) as page:
            if page.is_redirect:
                url, params = str(page.url.join(page.headers.get("location", ""))), None
                continue
            body = await read_capped(page, MAX_PAGE_BYTES)
            return page.status_code, body.decode(page.charset_encoding or "utf-8", errors="replace")
    raise ValueError(f"more than {MAX_REDIRECTS} redirects")


async def fetch_page(url: str, transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    if not url.startswith(("http://", "https://")):
        return {"error": "only http(s) URLs"}
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False, transport=transport) as http:
            status, text = await fetch(http, url)
    except ValueError as exc:
        return {"error": str(exc)}
    except httpx.HTTPError as exc:
        return {"error": f"could not fetch {url}: {exc}"}
    return {"status": status, "text": re.sub(r"<[^>]+>", " ", text)[:MAX_TEXT_CHARS]}


async def search_arxiv(query: str, max_results: int = 5,
                       transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    """Papers on arXiv matching `query`: title, authors, date, abstract (cut to 600 characters) and links."""
    params = {"search_query": f"all:{query}", "start": 0, "max_results": min(max(max_results, 1), 20)}
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=False, transport=transport) as http:
            status, text = await fetch(http, ARXIV, params)
        if status != 200:
            return {"error": f"arXiv returned HTTP {status}"}
        if re.search(r"<!(DOCTYPE|ENTITY)", text, re.IGNORECASE):  # Atom needs no DTD; entities can expand ~100x
            return {"error": "the arXiv response declared a DTD; refusing to parse it"}
        root = ET.fromstring(text)
    except ValueError as exc:
        return {"error": str(exc)}
    except (httpx.HTTPError, ET.ParseError) as exc:
        return {"error": f"arXiv search failed: {exc}"}
    papers = []
    for entry in root.findall("a:entry", ATOM):
        def field(tag: str, node: ET.Element = entry) -> str:
            return " ".join((node.findtext(f"a:{tag}", "", ATOM) or "").split())
        papers.append({
            "id": field("id"), "title": field("title"), "published": field("published")[:10],
            "authors": [" ".join((a.findtext("a:name", "", ATOM) or "").split())
                        for a in entry.findall("a:author", ATOM)],
            "abstract": field("summary")[:600],
            "pdf": next((link.get("href") for link in entry.findall("a:link", ATOM)
                         if link.get("title") == "pdf"), None),
        })
    return {"papers": papers}

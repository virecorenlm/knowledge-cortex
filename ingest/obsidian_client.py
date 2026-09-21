"""Async client for the Obsidian MCP server (streamable HTTP transport).

Reuses the same MCP endpoint/auth Hermes already talks to. Configuration is
read from environment variables at call time (not import time) so tests can
monkeypatch or inject a fake client.

Environment variables:
    OBSIDIAN_MCP_URL          Default: http://127.0.0.1:27124/mcp/
    OBSIDIAN_MCP_AUTHORIZATION  Authorization header value (e.g. "Bearer ...").
                                Required for a self-hosted Obsidian MCP server
                                with auth enabled.
"""

import json
import os

DEFAULT_URL = "http://127.0.0.1:27124/mcp/"


def _env(key, default=None):
    return os.getenv(key, default)


class ObsidianClient:
    """Wraps mcp's streamable HTTP client + ClientSession lifecycle so callers
    can await simple methods instead of managing the session themselves.
    """

    def __init__(self, url=None, authorization=None):
        self.url = url or _env("OBSIDIAN_MCP_URL", DEFAULT_URL) or DEFAULT_URL
        self.authorization = authorization if authorization is not None else _env("OBSIDIAN_MCP_AUTHORIZATION")

    def _headers(self):
        headers = {}
        if self.authorization:
            headers["Authorization"] = self.authorization
        return headers

    async def _call(self, tool_name, arguments):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client, create_mcp_http_client

        async with create_mcp_http_client(headers=self._headers()) as http:
            async with streamable_http_client(self.url, http_client=http) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
                    if result.is_error:
                        raise RuntimeError(f"Obsidian MCP tool '{tool_name}' failed: {text}")
                    return text

    async def list_dir(self, path=""):
        text = await self._call("vault_list", {"path": path} if path else {})
        return json.loads(text)["files"]

    async def read_note(self, path):
        text = await self._call("vault_read", {"path": path})
        return json.loads(text)

    async def write_note(self, path, content):
        return await self._call("vault_write", {"path": path, "content": content})

    async def append_note(self, path, content):
        return await self._call("vault_append", {"path": path, "content": content})

    async def iter_markdown_paths(self, root="", exclude_substrings=()):
        """Yield every markdown note path under root, walking subdirectories,
        skipping any path containing one of exclude_substrings (case-insensitive).
        """
        stack = [root]
        seen = set()
        while stack:
            directory = stack.pop()
            if directory in seen:
                continue
            seen.add(directory)
            entries = await self.list_dir(directory)
            for name in entries:
                full = f"{directory}/{name}".lstrip("/") if directory else name
                lowered = full.lower()
                if any(s in lowered for s in exclude_substrings):
                    continue
                if name.endswith("/"):
                    stack.append(full.rstrip("/"))
                elif name.endswith(".md"):
                    yield full

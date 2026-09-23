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


def _first_leaf(group):
    while isinstance(group, BaseExceptionGroup) and group.exceptions:
        group = group.exceptions[0]
    return group


class ObsidianClient:
    """Wraps mcp's streamable HTTP client + ClientSession lifecycle so callers
    can await simple methods instead of managing the session themselves.
    """

    # Rename/delete propagation needs no-clobber moves. The server's verified
    # vault_move(path, destination, allowOverwrite=false) refuses to clobber,
    # but it also rewrites internal links in OTHER notes that point at the
    # moved file (Obsidian rename semantics), a side effect outside the managed
    # subtree. Moves are therefore opt-in: allow_moves=True or
    # OBSIDIAN_ALLOW_MOVES=1; otherwise move-dependent operations fail closed.
    supports_moves = False

    def __init__(self, url=None, authorization=None, allow_moves=None):
        self.url = url or _env("OBSIDIAN_MCP_URL", DEFAULT_URL) or DEFAULT_URL
        self.authorization = authorization if authorization is not None else _env("OBSIDIAN_MCP_AUTHORIZATION")
        if allow_moves is None:
            allow_moves = (_env("OBSIDIAN_ALLOW_MOVES") or "").strip().lower() in ("1", "true", "yes")
        self.supports_moves = bool(allow_moves)

    def _headers(self):
        headers = {}
        if self.authorization:
            headers["Authorization"] = self.authorization
        return headers

    async def _call(self, tool_name, arguments):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client, create_mcp_http_client

        # Tool errors are raised only after the session has closed: raised
        # inside it, anyio's task groups wrap them in an ExceptionGroup, and
        # callers would never see the documented RuntimeError. Transport
        # failures are unwrapped the same way.
        try:
            async with create_mcp_http_client(headers=self._headers()) as http:
                async with streamable_http_client(self.url, http_client=http) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments)
                        text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
                        is_error = result.is_error
        except BaseExceptionGroup as group:
            leaf = _first_leaf(group)
            if isinstance(leaf, (KeyboardInterrupt, SystemExit)):
                raise leaf
            raise RuntimeError(f"Obsidian MCP call '{tool_name}' failed: {leaf}") from group
        if is_error:
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

    @staticmethod
    def _checked(path):
        if not isinstance(path, str) or not path or path.startswith("/") or "\x00" in path or "\\" in path \
                or any(part in ("", ".", "..") for part in path.split("/")):
            raise ValueError(f"unsafe vault path {path!r}")
        return path

    async def move_note(self, src, dst):
        """No-clobber move (server refuses if dst exists). Opt-in, see supports_moves."""
        if not self.supports_moves:
            raise RuntimeError("vault moves are disabled for this backend (set OBSIDIAN_ALLOW_MOVES=1)")
        return await self._call("vault_move", {"path": self._checked(src), "destination": self._checked(dst),
                                               "allowOverwrite": False})

    async def remove_note(self, path):
        """Used only to purge Cortex recovery copies. NOT a hard delete: the
        server moves the file to Obsidian's trash (.trash or the system trash,
        per the user's "Deleted files" setting)."""
        return await self._call("vault_delete", {"path": self._checked(path), "permanent": False})

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

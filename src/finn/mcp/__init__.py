"""MCP server integration via LangChain tools.

Loads from ``mcp_config.json``, converts MCP tools to LangChain ``BaseTool``
instances that can be passed to ``ChatOpenAI.bind_tools()``.

Usage::

    from finn.mcp import mcp_session

    async with mcp_session() as (tools, _):
        llm_with_tools = llm.bind_tools(tools)
        ...

Tool filtering via MCP_TOOL_FILTERS.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field, create_model

from finn.config import config
from finn.logger import logger

# ═══════════════════════════════════════════════════════════════════════
# Tool filters
# ═══════════════════════════════════════════════════════════════════════

MCP_TOOL_FILTERS: dict[str, dict] = {
    "amap-maps-streamableHTTP": {
        "include": {
            "maps_text_search",
            "maps_around_search",
            "maps_search_detail",
            "maps_geo",
            "maps_regeocode",
            "maps_distance",
            "maps_direction_driving",
            "maps_direction_walking",
            "maps_direction_bicycling",
            "maps_direction_transit_integrated",
            "maps_weather",
        },
    },
}


def _tool_allowed(tool_name: str, server_id: str) -> bool:
    rules = MCP_TOOL_FILTERS.get(server_id)
    if rules is None:
        return True
    if "include" in rules:
        return tool_name in rules["include"]
    if "exclude" in rules:
        return tool_name not in rules["exclude"]
    return True


# ═══════════════════════════════════════════════════════════════════════
# JSON Schema → Pydantic args_schema
# ═══════════════════════════════════════════════════════════════════════

# MCP tools receive all arguments as strings via JSON-RPC.
# Use str for everything to avoid type coercion issues (e.g. model sends
# {"type": 1} but schema expects {"type": "1"}).
_TYPE_MAP: dict[str, type] = {
    "string": str, "number": str, "integer": str,
    "boolean": str, "array": str, "object": str,
}


def _build_args_schema(schema: dict, name: str) -> type:
    """Convert JSON Schema to a Pydantic model for LangChain args_schema."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}

    for field_name, prop in props.items():
        py_type = _TYPE_MAP.get(prop.get("type", "string"), str)
        desc = prop.get("description", "")
        if field_name in required:
            fields[field_name] = (py_type, Field(description=desc))
        else:
            fields[field_name] = (py_type | None, Field(default=None, description=desc))

    if not fields:
        fields["query"] = (str, Field(description="Query string"))

    return create_model(f"Args_{name}", **fields)


# ═══════════════════════════════════════════════════════════════════════
# MCP → LangChain tool wrapper
# ═══════════════════════════════════════════════════════════════════════


class _MCPTool(BaseTool):
    """LangChain tool wrapping an MCP tool. Calls the MCP session directly."""

    session: Any = None

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async")

    async def _arun(self, **kwargs) -> str:
        try:
            result = await self.session.call_tool(self.name, kwargs)
            if hasattr(result, "content") and result.content:
                # MCP CallToolResult
                texts = []
                for c in result.content:
                    if hasattr(c, "text"):
                        texts.append(c.text)
                return "\n".join(texts)
            return str(result)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)


def _make_langchain_tools(session, server_id: str) -> list[BaseTool]:
    """List MCP tools via session, filter, convert to LangChain tools."""
    # Tools are listed during initialize; use internal state
    # We need to call list_tools via the session
    raise NotImplementedError("Use async _make_langchain_tools_async")


# ═══════════════════════════════════════════════════════════════════════
# MCP session context manager
# ═══════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def mcp_session(config_path: str | Path | None = None):
    """Connect to MCP servers, yield (langchain_tools, server_id).

    Handles the full MCP lifecycle: connect → initialize → list_tools → yield → cleanup.

    Yields ``(tools, server_id)`` where tools is a list of LangChain BaseTool
    instances that can be used with ``ChatOpenAI.bind_tools()``.
    """
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession

    resolved = str(config_path) if config_path else str(Path.cwd() / "mcp_config.json")

    if not Path(resolved).exists():
        logger.debug("MCP config not found at %s", resolved)
        yield ([], "")
        return

    with open(resolved) as f:
        mcp_config = json.load(f)

    servers = mcp_config.get("mcpServers", {})
    if not servers:
        logger.debug("No MCP servers configured")
        yield ([], "")
        return

    # Trigger .env loading for env var expansion
    _ = config.amap_api_key

    def _expand_env(value: str) -> str:
        def _replace(match):
            var_name = match.group(1)
            default = match.group(2)
            return os.environ.get(var_name, default or "")
        return re.sub(r'\$\{(\w+)(?::-([^}]*))?\}', _replace, value)

    for server_id, server_cfg in servers.items():
        url = _expand_env(server_cfg.get("url", ""))
        if not url:
            continue

        logger.info("MCP: connecting to %s", server_id)
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("MCP: %s initialized", server_id)

                # List tools from MCP server
                result = await session.list_tools()
                all_tools = result.tools

                # Filter and convert to LangChain tools
                tools: list[BaseTool] = []
                for t in all_tools:
                    if not _tool_allowed(t.name, server_id):
                        continue

                    input_schema = getattr(t, "inputSchema", {}) or {}
                    args_schema = _build_args_schema(input_schema, t.name)

                    langchain_tool = _MCPTool(
                        name=t.name,
                        description=t.description or f"MCP: {t.name}",
                        session=session,
                        args_schema=args_schema,
                    )
                    tools.append(langchain_tool)

                logger.info("MCP: %d tools loaded (filtered from %d)",
                           len(tools), len(all_tools))
                for t in tools:
                    logger.debug("  MCP: %s — %s", t.name, t.description[:80])

                yield (tools, server_id)
                return

    yield ([], "")


def reload_mcp_tools():
    """No-op — sessions are created fresh each time. Kept for API compat."""
    pass

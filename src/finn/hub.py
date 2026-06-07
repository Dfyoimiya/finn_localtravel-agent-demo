"""MCPHub — centralized TTL cache + batch helpers for MCP tool calls.

Each LangGraph node opens its own MCP session (see mcp/__init__.py for why).
MCPHub provides a process-level cache that survives session teardown, plus
parallel batch helpers that take the session's ``tools`` list as a parameter.

Usage::

    from finn.hub import mcp_hub

    async with mcp_session() as (tools, _):
        results = await mcp_hub.batch_around_search(tools, loc, keywords, radius)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

from finn.logger import logger


class MCPHub:
    """Process-level TTL cache + batch helpers for MCP tool calls.

    Does NOT manage sessions. Receives ``tools: list[BaseTool]`` from
    whichever MCP session the calling node opens.
    """

    def __init__(self, ttl_seconds: int = 300, max_concurrency: int = 3):
        self._store: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)

    # ── cache ──────────────────────────────────────────────────────

    def _make_key(self, *parts: str) -> str:
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)

    def clear(self) -> int:
        """Clear all cached entries. Returns count of cleared entries."""
        count = len(self._store)
        self._store.clear()
        logger.debug("MCPHub cache cleared (%d entries)", count)
        return count

    # ── batch helpers ──────────────────────────────────────────────

    async def batch_around_search(
        self, tools, location: str, keywords: list[str], radius: int,
    ) -> list[tuple[str, list[dict]]]:
        """Parallel maps_around_search for multiple keywords.

        Returns list of (keyword, raw_pois) tuples. Cached by (location, keyword, radius).
        """
        from finn.mcp import _MCPTool  # noqa: F811

        around_tool = next((t for t in tools if t.name == "maps_around_search"), None)
        if not around_tool:
            return []

        async def _search_one(kw: str):
            cache_key = self._make_key("around", location, kw, str(radius))
            cached = self.get(cache_key)
            if cached is not None:
                logger.debug("MCPHub cache HIT around_search(%s)", kw)
                return (kw, cached)

            try:
                async with self._semaphore:
                    result = await around_tool.ainvoke({
                        "location": location,
                        "keywords": kw,
                        "radius": str(radius),
                    })
                parsed = self._parse_pois(str(result))
                self.set(cache_key, parsed)
                return (kw, parsed)
            except Exception as exc:
                logger.debug("around_search(%s) failed: %s", kw, exc)
                return (kw, [])

        tasks = [_search_one(kw) for kw in keywords]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def batch_search_detail(
        self, tools, poi_ids: list[str],
    ) -> dict[str, dict]:
        """Parallel maps_search_detail for multiple POI IDs.

        Returns {poi_id: detail_dict}. Cached by poi_id.
        Uses semaphore (max 3 concurrent) to avoid Amap API rate limiting.
        """
        detail_tool = next((t for t in tools if t.name == "maps_search_detail"), None)
        if not detail_tool:
            return {}

        async def _detail_one(pid: str):
            cache_key = self._make_key("detail", pid)
            cached = self.get(cache_key)
            if cached is not None:
                return (pid, cached)

            try:
                async with self._semaphore:
                    result = await detail_tool.ainvoke({"id": pid})
                data = self._extract_json(str(result)) if "{" in str(result) else {}
                self.set(cache_key, data)
                return (pid, data)
            except Exception as exc:
                logger.debug("search_detail(%s) failed: %s", pid, exc)
                return (pid, {})

        tasks = [_detail_one(pid) for pid in poi_ids]
        results = await asyncio.gather(*tasks)
        return {pid: data for pid, data in results if data}

    async def batch_distance(
        self, tools, pairs: list[tuple[str, str]], mode: str = "0",
    ) -> dict[str, int]:
        """Parallel maps_distance for coordinate pairs.

        ``pairs`` is a list of (origin_lnglat, dest_lnglat).
        ``mode``: "0"=driving (default), "1"=walking, "2"=bus/transit.
        Returns {"origin|dest": distance_m}. Cached.
        """
        dist_tool = next((t for t in tools if t.name == "maps_distance"), None)
        if not dist_tool:
            return {}

        async def _dist_one(pair: tuple[str, str]):
            origin, dest = pair
            pair_key = f"{origin}|{dest}"
            cache_key = self._make_key("distance", pair_key, mode)
            cached = self.get(cache_key)
            if cached is not None:
                return (pair_key, cached)

            try:
                async with self._semaphore:
                    result = await dist_tool.ainvoke({
                        "origins": origin,
                        "destination": dest,
                        "type": mode,
                    })
                data = self._extract_json(str(result)) if "{" in str(result) else {}
                results_list = data.get("results", [])
                distance = int(results_list[0].get("distance", 0)) if results_list else 0
                self.set(cache_key, distance)
                return (pair_key, distance)
            except Exception as exc:
                logger.debug("distance(%s→%s) failed: %s", origin[:12], dest[:12], exc)
                return (pair_key, 0)

        tasks = [_dist_one(p) for p in pairs]
        results = await asyncio.gather(*tasks)
        return {k: v for k, v in results}

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _parse_pois(text: str) -> list[dict]:
        """Extract POI list from Amap around_search response text."""
        data = MCPHub._extract_json(text) if "{" in text else {}
        return data.get("pois", [])

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract first JSON object from text (greedy match)."""
        import re
        # Greedy match — handles any nesting depth
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}


# Module-level singleton
mcp_hub = MCPHub()

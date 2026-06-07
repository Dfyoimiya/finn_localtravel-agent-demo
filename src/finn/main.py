"""Finn — local short-trip planning agent.

Usage:
    uv run finn          # interactive multi-turn session
    uv run finn -v       # verbose LLM call logging (DEBUG level)
"""

from __future__ import annotations

import asyncio
import logging
import sys

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from finn.cli import CLI
from finn.config import config
from finn.graph import create_graph
from finn.logger import logger

# ═══════════════════════════════════════════════════════════════════════
# Checkpoint serialization — register custom types for msgpack
# ═══════════════════════════════════════════════════════════════════════

_checkpoint_serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("finn.state", "SceneType"),
        ("finn.state", "BudgetPreference"),
        ("finn.state", "TravelPace"),
        ("finn.state", "TransportMode"),
        ("finn.state", "ExtractResult"),
        ("finn.state", "UserIntent"),
        ("finn.state", "UserRequirements"),
        ("finn.state", "HardConstraints"),
        ("finn.state", "SoftConstraints"),
        ("finn.state", "GroupProfile"),
        ("finn.state", "TimeWindow"),
        ("finn.state", "GeoConstraint"),
        ("finn.state", "ChainTemplate"),
        ("finn.state", "UpdateExtractResultInput"),
        ("finn.state", "PartyMember"),
        ("finn.state", "POICandidate"),
        ("finn.state", "POISearchResult"),
        ("finn.state", "Plan"),
        ("finn.state", "Verification"),
        ("finn.state", "SubTask"),
        ("finn.state", "BookingResult"),
        ("finn.state", "WeatherContext"),
        ("finn.state", "PlanCard"),
        ("finn.state", "CheckerIssue"),
        ("finn.state", "ConstraintProfile"),
        ("finn.state", "TimeAlloc"),
        ("finn.state", "ActivityNode"),
        ("finn.state", "PlannedPath"),
        ("finn.state", "StrategyResult"),
        ("finn.state", "CategorySearch"),
        ("finn.state", "SearchStrategy"),
        ("finn.state", "POICategoryPool"),
        ("finn.state", "AgentPlan"),
        ("finn.state", "FusionResult"),
    ],
)
_checkpoint_saver = MemorySaver(serde=_checkpoint_serde)


# ═══════════════════════════════════════════════════════════════════════
# Callbacks — wired into the CLI; invoked by nodes via contextvars
# ═══════════════════════════════════════════════════════════════════════

def _on_token(token: str) -> None:
    """Streaming token callback — print each delta via Rich console."""
    cli = CLI._current_instance
    if cli is not None:
        cli._stream_token(token)
    else:
        sys.stdout.write(token)
        sys.stdout.flush()


def _on_tool(tool_name: str, tool_input: dict) -> None:
    """Tool call callback — display tool invocations during ReAct loops."""
    from rich.panel import Panel

    cli = CLI._current_instance
    if cli is not None:
        cli._console.print(
            Panel(
                f"[bold]Tool:[/bold] {tool_name}\n"
                f"```json\n{tool_input!r}\n```",
                border_style="yellow",
            )
        )


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════


async def run_interactive() -> None:
    """Set up logging, build the graph, and launch the CLI REPL."""
    # ── Logging ──
    log_level = logging.DEBUG if "-v" in sys.argv else logging.INFO
    logger.setup(level=log_level)

    # ── Graph ──
    graph = create_graph()
    graph.checkpointer = _checkpoint_saver

    # ── CLI ──
    cli = CLI(on_token=_on_token, on_tool=_on_tool)
    await cli.run(graph, _checkpoint_saver)


def main() -> None:
    asyncio.run(run_interactive())


if __name__ == "__main__":
    main()

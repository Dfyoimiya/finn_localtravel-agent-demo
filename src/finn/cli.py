"""CLI REPL — prompt_toolkit input loop + Rich Markdown rendering.

Usage:
    from finn.cli import CLI

    cli = CLI(on_token=..., on_tool=...)
    await cli.run(graph, checkpointer)

Callbacks
    ``on_token(token: str)``
        Called for each streaming text delta from the LLM.
        Pass ``None`` to disable streaming (nodes use plain ``agent.run()``).

    ``on_tool(tool_name: str, tool_input: dict)``
        Called when the agent invokes a tool during a ReAct loop.
        Pass ``None`` if tool display is not needed.

Both callbacks are threaded through the LangGraph nodes via
``contextvars`` so they don't need to appear in the state schema.
"""

from __future__ import annotations

import asyncio
import sys
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from finn.location import get_current_context, format_context
from finn.logger import logger
from finn.memory import MemoryManager, ProfileBuilder
from finn.memory.extractor import extract_learnings_from_trip
from finn.memory.models import TripMemory

# ═══════════════════════════════════════════════════════════════════════
# Context variables — set by CLI before graph invocation, read by nodes
# ═══════════════════════════════════════════════════════════════════════

_on_token_ctx: ContextVar[Callable[[str], None] | None] = ContextVar(
    "on_token", default=None
)
_on_tool_ctx: ContextVar[Callable[[str, dict], None] | None] = ContextVar(
    "on_tool", default=None
)


def get_on_token() -> Callable[[str], None] | None:
    """Return the current streaming callback, or None."""
    return _on_token_ctx.get(None)


def get_on_tool() -> Callable[[str, dict], None] | None:
    """Return the current tool callback, or None."""
    return _on_tool_ctx.get(None)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

_HISTORY_FILE = ".finn_history"

_STYLE = Style.from_dict({
    "prompt": "bold #00d7af",
    "separator": "#666666",
})


def _summarise_intent(intent) -> str:
    """Create a short Chinese summary of an Intent for trip memory."""
    parts = []
    if intent.scenario and intent.scenario != "unknown":
        labels = {"family": "家人", "friends": "朋友", "couple": "情侣", "solo": "独自"}
        parts.append(f"与{labels.get(intent.scenario, intent.scenario)}")
    if intent.party_size:
        parts.append(f"{intent.party_size}人")
    if intent.activity:
        parts.append(intent.activity)
    if intent.area:
        parts.append(f"在{intent.area}")
    if intent.budget_per_person:
        parts.append(f"人均{int(intent.budget_per_person)}元")
    return "，".join(parts)


class CLI:
    """Interactive REPL for Finn with Rich rendering and streaming support.

    Parameters
    ----------
    on_token:
        Streaming token callback. Receives each text delta.
        If ``None``, LLM calls use non-streaming mode.
    on_tool:
        Tool invocation callback. Receives ``(tool_name, tool_input)``.
    """

    _current_instance: CLI | None = None

    def __init__(
        self,
        on_token: Callable[[str], None] | None = None,
        on_tool: Callable[[str, dict], None] | None = None,
    ):
        self.on_token = on_token
        self.on_tool = on_tool

        self._console = Console()
        self._session = PromptSession(
            history=FileHistory(_HISTORY_FILE),
            style=_STYLE,
        )

        # Accumulator for streaming tokens during a graph invocation
        self._stream_buffer: list[str] = []
        self._streaming: bool = False

    # ── context management ──────────────────────────────────────────

    def _push_callbacks(self) -> None:
        """Set context vars so nodes can discover the callbacks."""
        CLI._current_instance = self
        _on_token_ctx.set(self.on_token)
        _on_tool_ctx.set(self.on_tool)

    def _stream_start(self) -> None:
        """Signal that a streaming LLM response is starting."""
        self._stream_buffer.clear()
        self._streaming = True

    def _stream_token(self, token: str) -> None:
        """Write a single token to the console during streaming."""
        self._console.print(token, end="", highlight=False)

    def _stream_end(self) -> None:
        """Signal that streaming has completed."""
        self._streaming = False
        if self._stream_buffer:
            self._stream_buffer.clear()

    # ── rendering helpers ───────────────────────────────────────────

    def print_markdown(self, text: str, title: str | None = None) -> None:
        """Render Markdown text inside an optional Rich Panel."""
        md = Markdown(text)
        if title:
            self._console.print(Panel(md, title=title, border_style="bold #00d7af"))
        else:
            self._console.print(md)

    def print_plan(self, plan_text: str) -> None:
        """Display a formatted plan for user review."""
        self.print_markdown(plan_text, title="Your Weekend Plan")

    def print_confirm_prompt(self, options: list[str]) -> str:
        """Display confirm/modify/cancel prompt and get user choice."""
        labels = {"confirm": "确认执行", "modify": "我要修改", "cancel": "算了"}
        prompt = " | ".join(
            f"[bold][{o}][/bold] {labels.get(o, o)}" for o in options
        )
        self._console.print(f"\n{prompt}\n")
        return ""  # caller handles input()

    def print_status(self, text: str) -> None:
        """Show a transient status message."""
        self._console.print(f"  {text}")

    # ── REPL loop ───────────────────────────────────────────────────

    async def run(
        self,
        graph,
        checkpointer,
        *,
        thread_id: str = "cli",
    ) -> None:
        """Start the interactive REPL.

        Parameters
        ----------
        graph:
            Compiled LangGraph StateGraph.
        checkpointer:
            LangGraph checkpointer (e.g. ``MemorySaver``).
        thread_id:
            Unique id for checkpoint isolation.
        """
        from langgraph.types import Command

        config = {"configurable": {"thread_id": thread_id}}

        self._console.print()
        self._console.print(
            Panel.fit(
                "输入出行需求，我来帮你规划 & 执行\n"
                '  "quit" 退出  |  "reset" 重置对话',
                title="Finn — 周末出行助手",
                border_style="bold #00d7af",
            )
        )
        self._console.print()

        # ── Cold-start profile check ──
        memory = MemoryManager()
        if not memory.profile_exists():
            builder = ProfileBuilder()
            profile = await builder.run(self)
            memory.save_profile(profile)
        else:
            # Load profile (triggers decay check) so it's cached
            memory.load_profile()

        is_tty = sys.stdin.isatty()

        while True:
            try:
                if is_tty:
                    user_input = await self._session.prompt_async(
                        [("class:prompt", "You: ")],
                    )
                else:
                    user_input = input("You: ")
            except (EOFError, KeyboardInterrupt):
                self._console.print("\n再见！")
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                self._console.print("再见！")
                break
            if user_input.lower() == "reset":
                config["configurable"]["thread_id"] = f"cli-{id(graph)}"
                self._console.print("[对话已重置]\n")
                continue
            if user_input.lower() == "/profile":
                profile = memory.load_profile()
                if not profile.setup_complete:
                    self._console.print("[尚未设置用户画像。]\n")
                else:
                    ctx_text = memory.build_profile_context()
                    self._console.print(
                        Panel(ctx_text, title="用户画像", border_style="bold #00d7af")
                    )
                    self._console.print()
                continue

            # ── Enrich with system context (time + IP location + profile) ──
            ctx = get_current_context()
            profile_ctx = memory.build_profile_context()
            enriched_input = (
                f"[系统上下文]\n{format_context(ctx)}\n{profile_ctx}\n[/系统上下文]\n\n{user_input}"
            )
            logger.debug("Context: %s", format_context(ctx))

            # ── Push callbacks & invoke graph ──
            self._push_callbacks()
            state = await graph.ainvoke(
                {"messages": [{"role": "user", "content": enriched_input}]},
                config,
            )

            # ── Cascading HITL interrupts ──
            interrupt_value = self._extract_interrupt(state)
            while interrupt_value and interrupt_value.get("type") == "plan_review":
                self.print_plan(interrupt_value["plan"])
                decision = self._handle_plan_review(interrupt_value)
                self._push_callbacks()
                state = await graph.ainvoke(Command(resume=decision), config)
                interrupt_value = self._extract_interrupt(state)

            # ── Display response ──
            self._display_response(state)

            # ── Post-trip learning ──
            self._learn_from_trip(state)

    # ── interrupt handling ──────────────────────────────────────────

    @staticmethod
    def _extract_interrupt(state: dict) -> dict | None:
        interrupts = state.get("__interrupt__")
        if not interrupts:
            return None
        return interrupts[0].value

    def _handle_plan_review(self, interrupt_value: dict):
        options = interrupt_value.get("options", ["confirm", "modify", "cancel"])
        labels = {"confirm": "确认执行", "modify": "我要修改", "cancel": "算了"}
        choice_parts = " | ".join(
            f"[{o}] {labels.get(o, o)}" for o in options
        )

        while True:
            choice = input(f"{choice_parts}\n> ").strip().lower()
            if choice == "modify":
                feedback = input("修改意见: ").strip()
                if feedback:
                    return {"action": "modify", "feedback": feedback}
                print("请输入修改意见")
            elif choice in options:
                return choice
            else:
                print(f"请输入: {' / '.join(options)}")

    # ── response display ────────────────────────────────────────────

    def _display_response(self, state: dict) -> None:
        intent = state.get("intent")
        next_action = state.get("next_action", "")

        if intent and intent.follow_up_question:
            self._console.print(f"\nFinn: {intent.follow_up_question}\n")
        elif next_action == "cancel":
            self._console.print("\nFinn: 好的，已取消。有需要随时找我。\n")
        elif state.get("messages"):
            last_msg = state["messages"][-1]
            role = getattr(last_msg, "type", "") or getattr(last_msg, "role", "")
            content = getattr(last_msg, "content", "")
            if role in ("ai", "assistant") and content:
                self._console.print()
                self.print_markdown(content)
                self._console.print()
        else:
            self._console.print("Finn: ...\n")

    # ── post-trip learning ──────────────────────────────────────────

    def _learn_from_trip(self, state: dict) -> None:
        """Extract preference signals from a completed trip and update profile."""
        import uuid
        from datetime import datetime, timezone, timedelta

        CST = timezone(timedelta(hours=8))

        intent = state.get("intent")
        execution_status = state.get("execution_status", "")

        # Only learn from trips that actually ran (not clarify rounds or cancels)
        if not intent or not intent.activity:
            return
        if intent.follow_up_question:
            return  # clarify round — no trip happened
        if execution_status not in ("done", "partial"):
            return

        plan = state.get("plan")
        memory = MemoryManager()

        # Extract preference signals
        learnings = extract_learnings_from_trip(intent, plan)
        if learnings:
            memory.update_preferences(learnings)
            logger.debug("Learned %d preference items from trip", len(learnings))

        # Build trip memory
        now = datetime.now(CST)
        trip_id = str(uuid.uuid4())[:8]
        trip = TripMemory(
            id=trip_id,
            created_at=now.isoformat(),
            intent_summary=_summarise_intent(intent),
            scenario=intent.scenario or "unknown",
            activity=intent.activity,
            date=intent.date,
            area=intent.area,
            party_size=intent.party_size,
            budget_total=intent.budget_total,
            plan_notes=plan.notes if plan else "",
            outcome="completed",
            extracted_learnings=learnings,
        )
        memory.save_trip(trip)

        # Update saved party members if new ones were described
        if intent.party_members:
            profile = memory.load_profile()
            if not profile.saved_party_members and intent.party_members:
                profile.saved_party_members = intent.party_members
                memory.save_profile(profile)

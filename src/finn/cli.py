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

            # ── Push callbacks & invoke graph ──
            self._push_callbacks()
            state = await graph.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
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

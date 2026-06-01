"""Finn — local short-trip planning agent.

Usage:
    uv run finn          # interactive multi-turn session
"""

import asyncio

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from finn.graph import create_graph

# Register custom types for checkpoint serialization
_checkpoint_serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("finn.state", "Intent"),
        ("finn.state", "Plan"),
        ("finn.state", "Verification"),
        ("finn.state", "SubTask"),
        ("finn.state", "BookingResult"),
    ],
)
_checkpoint_saver = MemorySaver(serde=_checkpoint_serde)


def _extract_interrupt(state: dict) -> dict | None:
    """If the graph paused on an interrupt, return its value."""
    interrupts = state.get("__interrupt__")
    if not interrupts:
        return None
    return interrupts[0].value


def _handle_plan_review(interrupt_value: dict):
    """Display the plan and collect user decision.

    Returns a value suitable for Command(resume=...):
      "confirm" / "cancel"
      {"action": "modify", "feedback": "user's modification request"}
    """
    print(f"\n{interrupt_value['plan']}\n")
    options = interrupt_value["options"]
    labels = {"confirm": "确认执行", "modify": "我要修改", "cancel": "算了"}
    prompt = " | ".join(f"[{o}] {labels.get(o, o)}" for o in options)

    while True:
        choice = input(f"{prompt}\n> ").strip().lower()
        if choice == "modify":
            feedback = input("修改意见: ").strip()
            if feedback:
                return {"action": "modify", "feedback": feedback}
            print("请输入修改意见")
        elif choice in options:
            return choice
        else:
            print(f"请输入: {' / '.join(options)}")


async def run_interactive():
    """Multi-turn CLI session with checkpointing and HITL interrupts."""
    load_dotenv()
    graph = create_graph()
    graph.checkpointer = _checkpoint_saver
    config = {"configurable": {"thread_id": f"cli-{id(graph)}"}}

    print("Finn — 周末出行助手")
    print('输入 "quit" 退出, "reset" 重置对话\n')

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见！")
            break
        if user_input.lower() == "reset":
            config["configurable"]["thread_id"] = f"cli-{id(graph)}"
            print("[对话已重置]\n")
            continue

        # ── Invoke the graph ──
        state = await graph.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
        )

        # ── Handle cascading interrupts (HITL) ──
        # modify → adjust → verify → present_to_user can trigger
        # a second interrupt. Loop until no more interrupts.
        interrupt_value = _extract_interrupt(state)
        while interrupt_value and interrupt_value.get("type") == "plan_review":
            decision = _handle_plan_review(interrupt_value)
            state = await graph.ainvoke(Command(resume=decision), config)
            interrupt_value = _extract_interrupt(state)

        # ── Display response ──
        intent = state.get("intent")
        next_action = state.get("next_action", "")

        if intent and intent.follow_up_question:
            print(f"Finn: {intent.follow_up_question}\n")
        elif next_action == "cancel":
            print("Finn: 好的，已取消。有需要随时找我。\n")
        elif state.get("messages"):
            last_msg = state["messages"][-1]
            if getattr(last_msg, "type", "") == "ai" or getattr(last_msg, "role", "") == "assistant":
                print(f"Finn: {last_msg.content}\n")
            # else: silent (e.g., user message echo)
        else:
            print("Finn: ...\n")


def main():
    asyncio.run(run_interactive())


if __name__ == "__main__":
    main()

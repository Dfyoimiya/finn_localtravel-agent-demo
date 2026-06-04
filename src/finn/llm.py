"""Shared LangChain LLM factory and ReAct helpers.

Replaces PydanticAI's Agent with LangChain's ChatOpenAI + manual ReAct loop.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler

from finn.config import config
from finn.logger import logger

# ═══════════════════════════════════════════════════════════════════════
# Model factory
# ═══════════════════════════════════════════════════════════════════════


def make_model(
    *,
    temperature: float = 0.7,
    streaming: bool = False,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance pointing at DeepSeek."""
    return ChatOpenAI(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        temperature=temperature,
        streaming=streaming,
        callbacks=callbacks or [],
    )


# ═══════════════════════════════════════════════════════════════════════
# Tool call logging
# ═══════════════════════════════════════════════════════════════════════

_COLOR_TOOL = "\033[38;5;178m"  # earth-yellow
_COLOR_RESET = "\033[0m"


def _log_tool_call(tool_name: str, args: dict, node: str) -> None:
    logger.info(
        "%s[Tool] %s | call %s(%s)%s",
        _COLOR_TOOL, node, tool_name,
        json.dumps(args, ensure_ascii=False),
        _COLOR_RESET,
    )


def _log_tool_return(tool_name: str, content: str, node: str) -> None:
    if len(content) > 300:
        content = content[:300] + "..."
    logger.info(
        "%s[Tool] %s | return %s → %s%s",
        _COLOR_TOOL, node, tool_name, content, _COLOR_RESET,
    )


# ═══════════════════════════════════════════════════════════════════════
# LLM call helpers
# ═══════════════════════════════════════════════════════════════════════


async def llm_invoke(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    node: str,
    *,
    stream: bool = False,
) -> str:
    """Invoke LLM and return response text, with timing and logging.

    When ``stream=True``, tokens are streamed to the CLI via ``on_token`` callback.
    """
    model_name = llm.model_name
    prompt_len = sum(len(str(m.content)) for m in messages)
    logger.info("→ %s | model=%s | prompt=%d chars", node, model_name, prompt_len)
    t0 = time.monotonic()

    try:
        if stream:
            from finn.cli import get_on_token
            on_token = get_on_token()
            if on_token:
                llm.streaming = True
                collected: list[str] = []
                async for chunk in llm.astream(messages):
                    if chunk.content:
                        on_token(chunk.content)
                        collected.append(chunk.content)
                text = "".join(collected)
            else:
                result = await llm.ainvoke(messages)
                text = result.content or ""
        else:
            result = await llm.ainvoke(messages)
            text = result.content or ""

        elapsed = time.monotonic() - t0
        logger.info("← %s | %d chars in %.1fs", node, len(text), elapsed)
        return text
    except Exception:
        elapsed = time.monotonic() - t0
        logger.error("✗ %s | failed after %.1fs", node, elapsed)
        raise


async def react_loop(
    llm: ChatOpenAI,
    tools: list[BaseTool],
    system_prompt: str,
    user_prompt: str,
    node: str,
    *,
    stream: bool = False,
) -> str:
    """Run a ReAct loop: LLM calls tools, gets results, continues until done.

    Returns the final text response (after all tool calls).
    """
    model_name = llm.model_name
    prompt_len = len(system_prompt) + len(user_prompt)
    logger.info("→ %s | model=%s | prompt=%d chars | tools=%d",
                 node, model_name, prompt_len, len(tools))
    t0 = time.monotonic()

    llm_with_tools = llm.bind_tools(tools)
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    try:
        while True:
            # ReAct loops always use non-streaming — streaming complicates
            # tool call accumulation across chunks.
            response = await llm_with_tools.ainvoke(messages)
            tool_calls = response.tool_calls
            final_text = response.content or ""

            if not tool_calls:
                # Done — no more tools to call
                break

            # Log and execute tool calls
            messages.append(response)
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                args = tc.get("args", {})
                _log_tool_call(tool_name, args, node)

                # Find and execute the tool
                tool = next((t for t in tools if t.name == tool_name), None)
                if tool:
                    try:
                        result = await tool.ainvoke(args)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = f"Tool '{tool_name}' not found"

                result_str = str(result)
                _log_tool_return(tool_name, result_str, node)
                messages.append(ToolMessage(
                    content=result_str,
                    tool_call_id=tc.get("id", ""),
                ))

        elapsed = time.monotonic() - t0
        logger.info("← %s | %d chars in %.1fs", node, len(final_text), elapsed)
        return final_text

    except Exception:
        elapsed = time.monotonic() - t0
        logger.error("✗ %s | failed after %.1fs", node, elapsed)
        raise

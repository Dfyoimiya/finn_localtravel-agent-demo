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


# Model families that only support temperature=1 (reasoning models)
_REASONING_MODEL_PREFIXES = ("kimi-k2", "deepseek-r1", "o1", "o3")


def make_model(
    *,
    temperature: float = 0.7,
    streaming: bool = False,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance pointing at the configured LLM provider.

    Auto-corrects temperature to 1 for reasoning models that don't support
    any other value (Kimi K2, DeepSeek-R1, OpenAI o1/o3, etc.).
    """
    model = config.llm_model.lower()
    if any(model.startswith(p) for p in _REASONING_MODEL_PREFIXES):
        if temperature != 1:
            logger.info("Model %s only supports temperature=1 — auto-corrected from %.1f",
                         config.llm_model, temperature)
            temperature = 1

    return ChatOpenAI(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        temperature=temperature,
        streaming=streaming,
        callbacks=callbacks or [],
    )


# ═══════════════════════════════════════════════════════════════════════
# Logging colors
# ═══════════════════════════════════════════════════════════════════════

_COLOR_TOOL = "\033[38;5;178m"   # earth-yellow
_COLOR_LLM = "\033[38;5;215m"    # Claude-style light orange
_COLOR_RESET = "\033[0m"

_MAX_CONTENT_LOG = 2000  # truncate logged content to this many chars


def _log_llm_prompt(node: str, messages: list[BaseMessage]) -> None:
    """Log full prompt content at DEBUG level in Claude-style orange."""
    logger.debug("%s┌─ Prompt (%s) ────────────────────────────%s", _COLOR_LLM, node, _COLOR_RESET)
    for i, msg in enumerate(messages):
        role = msg.__class__.__name__.replace("Message", "").upper()
        content = str(msg.content)
        if len(content) > _MAX_CONTENT_LOG:
            content = content[:_MAX_CONTENT_LOG] + f"\n... [truncated, total {len(str(msg.content))} chars]"
        logger.debug("%s[%s]%s %s", _COLOR_LLM, role, _COLOR_RESET, content)
    logger.debug("%s└────────────────────────────────────────────%s", _COLOR_LLM, _COLOR_RESET)


def _log_llm_response(node: str, text: str) -> None:
    """Log full LLM response at DEBUG level in Claude-style orange."""
    content = text if len(text) <= _MAX_CONTENT_LOG else text[:_MAX_CONTENT_LOG] + f"\n... [truncated, total {len(text)} chars]"
    logger.debug("%s┌─ Response (%s) ───────────────────────────%s", _COLOR_LLM, node, _COLOR_RESET)
    logger.debug("%s%s%s", _COLOR_LLM, content, _COLOR_RESET)
    logger.debug("%s└────────────────────────────────────────────%s", _COLOR_LLM, _COLOR_RESET)


# ═══════════════════════════════════════════════════════════════════════
# Tool call logging
# ═══════════════════════════════════════════════════════════════════════


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
    _log_llm_prompt(node, messages)
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
        _log_llm_response(node, text)
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
    max_tool_calls: int = 5,
) -> str:
    """Run a ReAct loop: LLM calls tools, gets results, continues until done.

    Returns the final text response (after all tool calls).

    Args:
        max_tool_calls: Maximum total tool invocations before forcing a stop.
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
    _log_llm_prompt(node, messages)

    total_tool_calls = 0

    try:
        while True:
            # ReAct loops always use non-streaming — streaming complicates
            # tool call accumulation across chunks.
            response = await llm_with_tools.ainvoke(messages)
            tool_calls = response.tool_calls
            final_text = response.content or ""

            if not tool_calls:
                break

            # Log intermediate response that triggered tool calls
            if final_text:
                _log_llm_response(f"{node}/turn", final_text)

            # Log and execute tool calls
            messages.append(response)
            for tc in tool_calls:
                total_tool_calls += 1
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

            if total_tool_calls >= max_tool_calls:
                logger.warning("react_loop: max_tool_calls (%d) reached — forcing stop", max_tool_calls)
                messages.append(HumanMessage(
                    content=f"已达到最大工具调用次数 ({max_tool_calls})。"
                            f"请基于已获取的信息直接输出最终 JSON 结果，不要再调用工具。"
                ))
                response = await llm.ainvoke(messages)
                final_text = response.content or ""
                break

        elapsed = time.monotonic() - t0
        logger.info("← %s | %d chars in %.1fs", node, len(final_text), elapsed)
        _log_llm_response(node, final_text)
        return final_text

    except Exception:
        elapsed = time.monotonic() - t0
        logger.error("✗ %s | failed after %.1fs", node, elapsed)
        raise


# ═══════════════════════════════════════════════════════════════════════
# Shared JSON extraction utility
# ═══════════════════════════════════════════════════════════════════════


def extract_json(text: str) -> dict:
    """Pull the first JSON object from LLM response text.

    Uses a balanced-brace approach to handle nested JSON correctly,
    as opposed to greedy regex which can match across multiple objects.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON found in response: {text[:200]}")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                json_str = text[start:i + 1]
                return json.loads(json_str)

    raise ValueError(f"Unbalanced braces in response: {text[:200]}")

"""PydanticAI ReAct agent with OpenAI-compatible LLM provider."""

import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


def create_agent() -> Agent:
    """Create a PydanticAI agent configured for ReAct loop."""
    provider = OpenAIProvider(
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        api_key=os.getenv("LLM_API_KEY"),
    )
    model = OpenAIModel(
        model_name=os.getenv("LLM_MODEL", "deepseek-chat"),
        provider=provider,
    )
    return Agent(
        model,
        system_prompt=(
            "You are Finn, a local short-trip planning assistant. "
            "Your goal is to help users get things done. "
            "When the user says hello, respond with exactly 'hello world'."
        ),
    )

"""Finn — local short-trip planning agent.

Usage:
    uv run finn
"""

import asyncio

from dotenv import load_dotenv

from finn.agent import create_agent
from finn.graph import create_graph


async def run():
    load_dotenv()
    agent = create_agent()
    graph = create_graph(agent)

    result = await graph.ainvoke({"input": "Say hello world"})
    print(result["output"])


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()

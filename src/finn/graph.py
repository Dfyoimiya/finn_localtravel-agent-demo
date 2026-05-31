"""LangGraph DAG — workflow orchestration layer.

DAG nodes are high-level workflow steps. Each node may internally run
a PydanticAI ReAct loop to produce its result.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph
from pydantic_ai import Agent


class AgentState(TypedDict):
    input: str
    output: str


def create_graph(agent: Agent):
    """Build a minimal DAG: START -> react_node -> END.

    In production this grows to:
        START -> intent -> plan -> execute* -> verify -> respond -> END
    where execute* nodes each run their own ReAct loop internally.
    """
    graph = StateGraph(AgentState)

    async def react_node(state: AgentState) -> dict:
        result = await agent.run(state["input"])
        return {"output": result.response.text}

    graph.add_node("react", react_node)
    graph.set_entry_point("react")
    graph.add_edge("react", END)

    return graph.compile()

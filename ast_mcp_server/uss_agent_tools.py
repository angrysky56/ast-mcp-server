"""
USS Agent tools for the MCP server.

This module exposes the USS Agent as MCP tools for natural language
interaction with the USS system.
"""

from typing import Any, Dict


def ask_uss_agent(query: str) -> Dict[str, Any]:
    """Natural language query to USS. Agent picks Neo4j/ChromaDB, runs query, returns {decision, results, summary}."""
    from ast_mcp_server.uss_agent import get_uss_agent

    try:
        agent = get_uss_agent()
        return agent.query_sync(query)
    except Exception as e:
        return {"error": str(e)}


def uss_agent_status() -> Dict[str, Any]:
    """Check USS Agent status: Neo4j connected?, ChromaDB available?, LLM model."""
    from ast_mcp_server.uss_agent import get_uss_agent

    try:
        agent = get_uss_agent()
        return agent.get_status()
    except Exception as e:
        return {"error": str(e)}


def register_uss_agent_tools(mcp_server: Any) -> None:
    """Register USS Agent tools with the MCP server."""
    mcp_server.tool()(ask_uss_agent)
    mcp_server.tool()(uss_agent_status)

"""
USS Agent tools for the MCP server.

This module exposes the USS Agent as MCP tools for natural language
interaction with the USS system.
"""

from typing import Any, Dict


def ask_uss_agent(query: str) -> Dict[str, Any]:
    """
    Ask the USS Agent a question about the codebase.

    The agent has access to both ChromaDB (semantic search) and Neo4j (graph DB),
    and can intelligently determine how to answer your question using the USS framework.

    Args:
        query: Natural language question or command

    Returns:
        Structured response with decision, results, and summary
    """
    from ast_mcp_server.uss_agent import get_uss_agent

    try:
        agent = get_uss_agent()
        return agent.query_sync(query)
    except Exception as e:
        return {"error": str(e)}


def uss_agent_status() -> Dict[str, Any]:
    """
    Get the status of the USS Agent and its database connections.

    Returns:
        Status of Neo4j, ChromaDB, and the configured LLM model
    """
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

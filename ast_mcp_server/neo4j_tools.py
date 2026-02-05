"""
Neo4j integration tools for the MCP server.

This module provides tools for syncing code analysis results (AST, ASG, Metrics)
to a Neo4j graph database.
"""

from typing import Any, Dict, Optional

from ast_mcp_server.neo4j_client import get_neo4j_client
from ast_mcp_server.tools import (
    analyze_code_structure,
    create_asg_from_ast,
    parse_code_to_ast,
)


def sync_file_to_graph(
    code: str, file_path: str, language: Optional[str] = None, project_name: Optional[str] = None
) -> Dict[str, Any]:
    """Parse code → store AST+ASG+metrics in Neo4j. Returns {stored: {ast_id, asg_id, analysis_id}}."""
    neo4j_client = get_neo4j_client()

    # Ensure connected and indexes present
    if not neo4j_client.is_connected():
        return {
            "error": "Neo4j connection not available. Check server logs and configuration."
        }

    # Run index creation (idempotent)
    neo4j_client.ensure_indexes()

    results: Dict[str, Any] = {
        "file_path": file_path,
        "status": "partial_success",
        "stored": {},
    }

    # 1. Parse AST
    ast_result = parse_code_to_ast(code, language=language, filename=file_path)
    if "error" in ast_result:
        return {"error": f"Failed to parse AST: {ast_result['error']}"}

    # Store AST
    ast_id = neo4j_client.store_ast(ast_result, file_path, project_name=project_name)
    if ast_id:
        results["stored"]["ast_id"] = ast_id

    # 2. Generate ASG
    asg_result = create_asg_from_ast(ast_result)
    if "error" not in asg_result:
        asg_id = neo4j_client.store_asg(asg_result, file_path, project_name=project_name)
        if asg_id:
            results["stored"]["asg_id"] = asg_id

    # 3. Analyze Structure
    analysis_result = analyze_code_structure(
        code, language=language, filename=file_path
    )
    if "error" not in analysis_result:
        analysis_id = neo4j_client.store_analysis(analysis_result, file_path, project_name=project_name)
        if analysis_id:
            results["stored"]["analysis_id"] = analysis_id

    results["status"] = "success"
    return results


def query_neo4j_graph(
    query: str, parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute Cypher query on code graph. Returns {records, count}."""
    neo4j_client = get_neo4j_client()

    if not neo4j_client.is_connected() or not neo4j_client.driver:
        return {
            "error": "Neo4j connection not available. Check server logs and configuration."
        }

    try:
        with neo4j_client.driver.session(database=neo4j_client.db) as session:
            result = session.run(query, parameters or {})
            records = [dict(record) for record in result]
            return {"records": records, "count": len(records)}
    except Exception as e:
        return {"error": f"Query execution failed: {str(e)}"}


def register_neo4j_tools(mcp_server: Any) -> None:
    """Register Neo4j tools with the MCP server."""
    mcp_server.tool()(sync_file_to_graph)
    mcp_server.tool()(query_neo4j_graph)

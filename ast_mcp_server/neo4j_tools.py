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


def register_neo4j_tools(mcp_server: Any) -> None:
    """Register Neo4j tools with the MCP server."""
    neo4j_client = get_neo4j_client()

    @mcp_server.tool()
    def sync_file_to_graph(
        code: str, file_path: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sync file analysis (AST, ASG, Metrics) to Neo4j graph.

        Args:
            code: Source code content
            file_path: Path to the file (used as ID/label)
            language: Programming language identifier

        Returns:
            Dictionary with IDs of stored artifacts or error message
        """
        # Ensure connected
        if not neo4j_client.is_connected():
            return {
                "error": "Neo4j connection not available. Check server logs and configuration."
            }

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
        ast_id = neo4j_client.store_ast(ast_result, file_path)
        if ast_id:
            results["stored"]["ast_id"] = ast_id

        # 2. Generate ASG
        asg_result = create_asg_from_ast(ast_result)
        if "error" not in asg_result:
            asg_id = neo4j_client.store_asg(asg_result, file_path)
            if asg_id:
                results["stored"]["asg_id"] = asg_id

        # 3. Analyze Structure
        analysis_result = analyze_code_structure(
            code, language=language, filename=file_path
        )
        if "error" not in analysis_result:
            analysis_id = neo4j_client.store_analysis(analysis_result, file_path)
            if analysis_id:
                results["stored"]["analysis_id"] = analysis_id

        results["status"] = "success"
        return results

    @mcp_server.tool()
    def query_graph(
        query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a Cypher query against the code graph.

        Args:
            query: Cypher query string
            parameters: Optional dictionary of query parameters

        Returns:
            List of records or error message
        """
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

"""
USS MCP Tools - Universal Semantic Structure tool registration.

Provides semantic search and graph query tools.
"""

from typing import Any, Dict, List, Optional

# Try to import USS components
try:
    from ast_mcp_server.uss_core import UniversalGraph
    from ast_mcp_server.vector_store import CHROMADB_AVAILABLE, get_vector_store

    USS_CORE_AVAILABLE = True
except ImportError:
    USS_CORE_AVAILABLE = False
    CHROMADB_AVAILABLE = False


def register_uss_tools(mcp_server: Any) -> None:
    """Register USS tools with the MCP server."""

    if not USS_CORE_AVAILABLE:
        return

    @mcp_server.tool()
    def semantic_search(
        query: str,
        node_type: Optional[str] = None,
        project: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search indexed content semantically. Returns matching nodes with relevance scores."""
        if not CHROMADB_AVAILABLE:
            return {
                "error": "ChromaDB not available. Install with: pip install chromadb"
            }

        store = get_vector_store()
        filters = {}
        if node_type:
            filters["type"] = node_type
        if project:
            filters["project"] = project

        results = store.search(
            query, n_results=limit, filters=filters if filters else None
        )

        return {
            "query": query,
            "count": len(results),
            "results": results,
        }

    @mcp_server.tool()
    def query_graph(
        graph_path: str,
        mode: str = "summary",
        node_id: Optional[str] = None,
        node_type: Optional[str] = None,
        edge_types: Optional[List[str]] = None,
        depth: int = 3,
    ) -> Dict[str, Any]:
        """Query nodes from saved graph by type/relationships. Returns matching nodes.

        Modes: summary, node, traverse, query
        - summary: Get node/edge counts and type distributions
        - node: Get full details for node_id (include connected edges)
        - traverse: Follow edge_types from node_id up to depth
        - query: Filter nodes by node_type
        """
        try:
            graph = UniversalGraph.load_json(graph_path)
        except Exception as e:
            return {"error": f"Failed to load graph: {e}"}

        if mode == "summary":
            return graph.get_summary()

        if mode == "node":
            if not node_id:
                return {"error": "node_id required for mode='node'"}
            node = graph.get_node(node_id)
            if not node:
                return {"error": f"Node {node_id} not found"}
            result = node.to_dict()
            result["edges"] = [e.to_dict() for e in graph.get_edges_for(node_id)]
            return result

        if mode == "traverse":
            if not node_id or not edge_types:
                return {"error": "node_id and edge_types required for mode='traverse'"}
            paths = graph.traverse(node_id, edge_types, depth)
            return {
                "start": node_id,
                "edge_types": edge_types,
                "depth": depth,
                "path_count": len(paths),
                "paths": [
                    [
                        {"id": n.id, "type": n.type, "preview": n.preview(50)}
                        for n in path
                    ]
                    for path in paths
                ],
            }

        if mode == "query":
            nodes = graph.query(node_type=node_type)
            return {
                "node_type": node_type,
                "count": len(nodes),
                "nodes": [
                    {"id": n.id, "type": n.type, "preview": n.preview(100)}
                    for n in nodes[:50]
                ],
            }

        return {"error": f"Unknown mode: {mode}. Use: summary, node, traverse, query"}


# CLI utility (not an MCP tool)
def check_uss_status() -> Dict[str, Any]:
    """Check USS component availability (CLI utility)."""
    try:
        import ast_mcp_server.embeddings  # noqa: F401

        embeddings_available = True
    except (ImportError, ValueError):
        embeddings_available = False

    try:
        import ast_mcp_server.server_llm  # noqa: F401

        server_llm_available = True
    except (ImportError, ValueError):
        server_llm_available = False

    return {
        "uss_core": USS_CORE_AVAILABLE,
        "chromadb": CHROMADB_AVAILABLE,
        "embeddings": embeddings_available,
        "server_llm": server_llm_available,
    }

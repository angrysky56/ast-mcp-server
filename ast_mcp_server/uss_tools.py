"""
USS MCP Tools - Universal Semantic Structure tool registration.

Provides MCP tools for semantic search, graph queries, and traversal.
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

try:
    import ast_mcp_server.embeddings  # noqa: F401

    EMBEDDINGS_AVAILABLE = True
except (ImportError, ValueError):
    EMBEDDINGS_AVAILABLE = False

try:
    import ast_mcp_server.server_llm  # noqa: F401

    SERVER_LLM_AVAILABLE = True
except (ImportError, ValueError):
    SERVER_LLM_AVAILABLE = False


def register_uss_tools(mcp_server: Any) -> None:
    """Register USS tools with the MCP server."""

    @mcp_server.tool()
    def uss_status() -> Dict[str, Any]:
        """Check USS (Universal Semantic Structure) component availability.

        Returns status of embeddings, vector store, and server LLM.
        """
        return {
            "uss_core": USS_CORE_AVAILABLE,
            "chromadb": CHROMADB_AVAILABLE,
            "embeddings": EMBEDDINGS_AVAILABLE,
            "server_llm": SERVER_LLM_AVAILABLE,
            "message": "USS components status. Set OPENROUTER_API_KEY for full functionality.",
        }

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
    def get_graph_node(
        node_id: str,
        include_edges: bool = True,
        graph_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get full node details from a saved graph. Optionally include edges."""
        if not graph_path:
            return {"error": "graph_path is required"}

        try:
            graph = UniversalGraph.load_json(graph_path)
        except Exception as e:
            return {"error": f"Failed to load graph: {e}"}

        node = graph.get_node(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}

        result = node.to_dict()

        if include_edges:
            result["edges"] = [e.to_dict() for e in graph.get_edges_for(node_id)]

        return result

    @mcp_server.tool()
    def traverse_graph(
        start_id: str,
        edge_types: List[str],
        depth: int = 3,
        graph_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Traverse graph from start node following specified edge types."""
        if not graph_path:
            return {"error": "graph_path is required"}

        try:
            graph = UniversalGraph.load_json(graph_path)
        except Exception as e:
            return {"error": f"Failed to load graph: {e}"}

        paths = graph.traverse(start_id, edge_types, depth)

        return {
            "start": start_id,
            "edge_types": edge_types,
            "depth": depth,
            "path_count": len(paths),
            "paths": [
                [{"id": n.id, "type": n.type, "preview": n.preview(50)} for n in path]
                for path in paths
            ],
        }

    @mcp_server.tool()
    def query_graph(
        node_type: Optional[str] = None,
        edge_type: Optional[str] = None,
        graph_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query nodes from saved graph by type/relationships. Returns matching nodes."""
        if not graph_path:
            return {"error": "graph_path is required"}

        try:
            graph = UniversalGraph.load_json(graph_path)
        except Exception as e:
            return {"error": f"Failed to load graph: {e}"}

        nodes = graph.query(node_type=node_type, edge_type=edge_type)

        return {
            "node_type": node_type,
            "edge_type": edge_type,
            "count": len(nodes),
            "nodes": [
                {"id": n.id, "type": n.type, "preview": n.preview(100)}
                for n in nodes[:50]  # Limit to avoid huge responses
            ],
        }

    @mcp_server.tool()
    def graph_summary(graph_path: Optional[str] = None) -> Dict[str, Any]:
        """Get node/edge counts and type distributions from saved graph."""
        if not graph_path:
            return {"error": "graph_path is required"}

        try:
            graph = UniversalGraph.load_json(graph_path)
        except Exception as e:
            return {"error": f"Failed to load graph: {e}"}

        return graph.get_summary()

"""
Vector Store module for Universal Semantic Structure.

Provides ChromaDB integration for semantic search over nodes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from ast_mcp_server.uss_core import UniversalNode

# Default storage path
DEFAULT_STORAGE_PATH = Path(__file__).parent.parent / "uss_data" / "chroma"


class VectorStore:
    """Vector store for semantic search over UniversalNodes using ChromaDB."""

    def __init__(
        self,
        path: Optional[str] = None,
        collection_name: str = "uss_nodes",
    ) -> None:
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )

        storage_path = Path(path) if path else DEFAULT_STORAGE_PATH
        storage_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(storage_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_nodes(self, nodes: List[UniversalNode], project: str = "default") -> int:
        """Add nodes to the vector store.

        Args:
            nodes: List of nodes with embeddings
            project: Project identifier for filtering

        Returns:
            Number of nodes added
        """
        # Filter to nodes with embeddings
        nodes_with_embeddings = [n for n in nodes if n.embedding is not None]

        if not nodes_with_embeddings:
            return 0

        self.collection.add(
            ids=[n.id for n in nodes_with_embeddings],
            embeddings=[n.embedding for n in nodes_with_embeddings],
            documents=[n.content for n in nodes_with_embeddings],
            metadatas=[
                {
                    "type": n.type,
                    "project": project,
                    **{
                        k: str(v)
                        for k, v in n.metadata.items()
                        if isinstance(v, (str, int, float, bool))
                    },
                }
                for n in nodes_with_embeddings
            ],
        )

        return len(nodes_with_embeddings)

    def search(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar nodes.

        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Optional metadata filters {"type": "function", ...}
            embedding: Optional pre-computed query embedding

        Returns:
            List of matching nodes with scores
        """
        # Build where clause from filters
        where = None
        if filters:
            if len(filters) == 1:
                key, value = list(filters.items())[0]
                where = {key: value}
            else:
                where = {"$and": [{k: v} for k, v in filters.items()]}

        # Search by query text or embedding
        if embedding:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        # Format results
        output = []
        if results["ids"] and results["ids"][0]:
            for i, node_id in enumerate(results["ids"][0]):
                output.append(
                    {
                        "id": node_id,
                        "content": (
                            results["documents"][0][i] if results["documents"] else ""
                        ),
                        "metadata": (
                            results["metadatas"][0][i] if results["metadatas"] else {}
                        ),
                        "distance": (
                            results["distances"][0][i] if results["distances"] else 0
                        ),
                        "score": 1.0
                        - (results["distances"][0][i] if results["distances"] else 0),
                    }
                )

        return output

    def delete_project(self, project: str) -> None:
        """Delete all nodes for a project."""
        self.collection.delete(where={"project": project})

    def count(self) -> int:
        """Get total number of nodes in store."""
        return int(self.collection.count())


# Module-level singleton
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

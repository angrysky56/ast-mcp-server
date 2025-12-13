"""
Universal Semantic Structure - Core Data Structures

Provides UniversalNode, SemanticEdge, and UniversalGraph for building
semantic graphs over any structured content (code, text, embeddings).
"""

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UniversalNode:
    """A node in the Universal Semantic Graph.

    Can represent code constructs, text segments, or embedding clusters.
    """
    id: str
    type: str  # "function", "class", "sentence", "cluster", etc.
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "children": self.children,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalNode":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=data["type"],
            content=data["content"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            children=data.get("children", []),
        )

    def preview(self, max_len: int = 100) -> str:
        """Get a short preview of the content."""
        if len(self.content) <= max_len:
            return self.content
        return self.content[:max_len] + "..."


@dataclass
class SemanticEdge:
    """An edge connecting two nodes in the semantic graph.

    Types include: calls, contains, similar_to, references, flows_to, etc.
    Weight indicates confidence (1.0 = definite, 0.5 = inferred).
    """
    source: str  # Node ID
    target: str  # Node ID
    type: str    # Edge type
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticEdge":
        """Create from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            type=data["type"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )


class UniversalGraph:
    """A semantic graph of nodes and edges.

    Supports structural queries, semantic search, and graph traversal.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, UniversalNode] = {}
        self.edges: List[SemanticEdge] = []
        self.index: Dict[str, List[str]] = defaultdict(list)  # type -> node_ids
        self._edge_index: Dict[str, List[SemanticEdge]] = defaultdict(list)  # node_id -> edges

    def add_node(self, node: UniversalNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self.index[node.type].append(node.id)

    def add_edge(self, edge: SemanticEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        self._edge_index[edge.source].append(edge)
        self._edge_index[edge.target].append(edge)

    def get_node(self, node_id: str) -> Optional[UniversalNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_edges_for(self, node_id: str) -> List[SemanticEdge]:
        """Get all edges connected to a node."""
        return self._edge_index.get(node_id, [])

    def query(
        self,
        node_type: Optional[str] = None,
        edge_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[UniversalNode]:
        """Query nodes by type and filters.

        Args:
            node_type: Filter by node type
            edge_type: Filter to nodes that have edges of this type
            filters: Additional metadata filters

        Returns:
            List of matching nodes
        """
        # Start with all nodes or filter by type
        if node_type:
            node_ids = self.index.get(node_type, [])
            candidates = [self.nodes[nid] for nid in node_ids]
        else:
            candidates = list(self.nodes.values())

        # Filter by edge type if specified
        if edge_type:
            nodes_with_edge = set()
            for edge in self.edges:
                if edge.type == edge_type:
                    nodes_with_edge.add(edge.source)
                    nodes_with_edge.add(edge.target)
            candidates = [n for n in candidates if n.id in nodes_with_edge]

        # Apply metadata filters
        if filters:
            result = []
            for node in candidates:
                match = True
                for key, value in filters.items():
                    if node.metadata.get(key) != value:
                        match = False
                        break
                if match:
                    result.append(node)
            candidates = result

        return candidates

    def traverse(
        self,
        start: str,
        edge_types: List[str],
        depth: int = 3,
    ) -> List[List[UniversalNode]]:
        """Traverse the graph following specific edge types.

        Args:
            start: Starting node ID
            edge_types: Edge types to follow
            depth: Maximum traversal depth

        Returns:
            List of paths (each path is a list of nodes)
        """
        paths: List[List[UniversalNode]] = []
        visited: set = set()

        def _traverse(node_id: str, current_path: List[UniversalNode], current_depth: int) -> None:
            if current_depth > depth or node_id in visited:
                return

            node = self.get_node(node_id)
            if not node:
                return

            visited.add(node_id)
            new_path = current_path + [node]

            # Find outgoing edges of specified types
            outgoing = [e for e in self._edge_index.get(node_id, [])
                       if e.source == node_id and e.type in edge_types]

            if not outgoing:
                # End of path
                if len(new_path) > 1:
                    paths.append(new_path)
            else:
                for edge in outgoing:
                    _traverse(edge.target, new_path, current_depth + 1)

            visited.remove(node_id)

        _traverse(start, [], 0)
        return paths

    def get_summary(self) -> Dict[str, Any]:
        """Get a lightweight summary of the graph."""
        edge_type_counts: Dict[str, int] = defaultdict(int)
        for edge in self.edges:
            edge_type_counts[edge.type] += 1

        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_types": {t: len(ids) for t, ids in self.index.items()},
            "edge_types": dict(edge_type_counts),
            "symbols": [
                {"id": n.id, "type": n.type, "preview": n.preview(50)}
                for n in list(self.nodes.values())[:20]
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire graph to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalGraph":
        """Create graph from dictionary."""
        graph = cls()
        for node_data in data.get("nodes", []):
            graph.add_node(UniversalNode.from_dict(node_data))
        for edge_data in data.get("edges", []):
            graph.add_edge(SemanticEdge.from_dict(edge_data))
        return graph

    def save_json(self, path: str) -> None:
        """Save graph to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "UniversalGraph":
        """Load graph from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def generate_node_id(prefix: str = "node") -> str:
    """Generate a unique node ID."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

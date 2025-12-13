# Universal Semantic Structure System (USS)

## A Self-Contained Blueprint for AST/ASG Applied to Any Structured Content

---

## 1. Core Concept

**Traditional View**: AST (Abstract Syntax Tree) parses programming languages.

**This System**: AST/ASG patterns apply to ANY structured content - code, embeddings, natural language, knowledge graphs. The parsing grammar is either explicit (tree-sitter) or learned (embeddings).

### Key Insight

Embeddings ARE a language. Vector spaces have structure:
- Clusters = "tokens/symbols"
- Distances = "syntax rules"  
- Directions = "semantic relationships"

An ASG built over embeddings creates a **navigable semantic graph** where AI can traverse meaning, not just similarity.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Universal Parser Layer                    │
├─────────────────────────────────────────────────────────────┤
│  Code Parser        │  Embedding Parser   │  NLP Parser      │
│  (tree-sitter)      │  (vector clustering)│  (sentence AST)  │
└─────────────────────┴─────────────────────┴──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Abstract Semantic Graph (ASG)                │
│  Nodes: Concepts/Symbols/Clusters                           │
│  Edges: Relationships (calls, contains, similar_to, etc.)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Query Interface                         │
│  - Structural queries (find all X that contain Y)           │
│  - Semantic queries (find concepts similar to Z)            │
│  - Hybrid queries (semantic search + structural filter)     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Data Structures

### 3.1 Universal Node

```python
@dataclass
class UniversalNode:
    id: str                          # Unique identifier
    type: str                        # "function", "cluster", "sentence", etc.
    content: str                     # Original text/code
    embedding: Optional[List[float]] # Vector representation
    metadata: Dict[str, Any]         # Source, line, confidence, etc.
    children: List[str]              # Child node IDs
```

### 3.2 Semantic Edge

```python
@dataclass
class SemanticEdge:
    source: str      # Node ID
    target: str      # Node ID
    type: str        # "calls", "contains", "similar_to", "references", "flows_to"
    weight: float    # Strength (1.0 = definite, 0.5 = inferred)
    metadata: Dict   # How this edge was derived
```

### 3.3 Universal Graph

```python
class UniversalGraph:
    nodes: Dict[str, UniversalNode]
    edges: List[SemanticEdge]
    index: Dict[str, List[str]]  # type -> node_ids for fast lookup
    
    def query(self, node_type: str = None, 
              edge_type: str = None,
              semantic_query: str = None,
              filters: Dict = None) -> List[UniversalNode]:
        """Hybrid structural + semantic query."""
        pass
    
    def traverse(self, start: str, edge_types: List[str], 
                 depth: int = 3) -> List[List[UniversalNode]]:
        """Graph traversal following specific edge types."""
        pass
```

---

## 4. Parser Implementations

### 4.1 Code Parser (tree-sitter)

Converts source code to AST, then enriches with semantic edges.

### 4.2 Embedding Parser (Novel)

Treats a vector space as a parseable structure:
1. Cluster vectors to find "tokens" (concepts)
2. Create nodes for clusters and items
3. Create edges: membership, similarity, nearest neighbors

### 4.3 NLP Parser

Parses natural language into structured AST using dependency parsing.

---

## 5. Query Interface

- **Structural Queries**: Find by type, filter by edges
- **Traversal Queries**: Follow edge chains through graph
- **Hybrid Queries**: Semantic search + structural filter

---

## 6. Storage Layer

### 6.1 Vector Store (ChromaDB)
For semantic search over embeddings.

### 6.2 Graph Store (JSON Files)
- index.json: Lightweight manifest
- Chunked by node type for targeted retrieval

---

## 7. Server LLM Integration

Offload processing from client AI:
- Summarize graph structures
- Answer queries with context
- Return lightweight responses

---

## 8. MCP Tool Interface

```python
@mcp.tool()
def parse_content(content: str, content_type: str = "auto") -> Dict:
    """Parse any content into Universal Graph. Returns lightweight index."""

@mcp.tool()
def semantic_search(query: str, filters: Dict = None) -> List[Dict]:
    """Search across all indexed content semantically."""

@mcp.tool()
def get_node(node_id: str, include_edges: bool = True) -> Dict:
    """Get full details for a specific node."""

@mcp.tool()
def traverse_graph(start_id: str, edge_types: List[str], depth: int = 3) -> List[Dict]:
    """Traverse relationships from a starting node."""

@mcp.tool()
def explain(node_id: str) -> str:
    """Get server LLM explanation of a node."""
```

---

## 9. Implementation Checklist

1. **Core Data Structures**: UniversalNode, SemanticEdge, UniversalGraph
2. **Parsers**: Code (tree-sitter), Text (spacy), Embedding (clustering)
3. **Storage**: ChromaDB vectors, JSON graph with chunking
4. **Server LLM**: OpenRouter integration
5. **MCP Interface**: All tools listed above

---

## 10. Dependencies

```toml
[project]
dependencies = [
    "mcp[cli]>=1.0.0",
    "tree-sitter>=0.25.0",
    "chromadb>=0.4.0",
    "httpx>=0.25.0",
    "spacy>=3.7.0",
    "hdbscan>=0.8.0",
    "numpy>=1.24.0",
]
```

---

## 11. Key Principles

1. **Everything is parseable** - Code, text, vectors all become graphs
2. **Lightweight to client, rich in storage** - Return summaries, store details
3. **Hybrid queries** - Combine structural + semantic
4. **Server does the heavy lifting** - LLM summarization happens server-side
5. **Traversable relationships** - Not just search, but navigation

---

*This document is self-contained. An AI reading only this file has everything needed to implement the system from scratch.*

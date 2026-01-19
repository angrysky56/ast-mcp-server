"""
USS Agent - Intelligent agent for the Universal Semantic Structure system.

This agent has access to both ChromaDB (vector store) and Neo4j (graph DB),
and can intelligently operate on the USS framework based on natural language queries.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Load .env from project root
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

# Import database clients
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ast_mcp_server.neo4j_client import Neo4jClient
    from ast_mcp_server.vector_store import VectorStore

_get_neo4j_client: Optional[Callable[[], "Neo4jClient"]] = None
_get_vector_store: Optional[Callable[[], "VectorStore"]] = None

try:
    from ast_mcp_server.neo4j_client import get_neo4j_client as _get_neo4j_fn

    _get_neo4j_client = _get_neo4j_fn
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    from ast_mcp_server.vector_store import get_vector_store as _get_vector_fn

    _get_vector_store = _get_vector_fn
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# System prompt that teaches the agent about USS
USS_SYSTEM_PROMPT = """You are the USS Agent - the intelligent curator of the Universal Semantic Structure system.

## Your Role
You manage and organize code knowledge across two complementary databases:
1. **ChromaDB (Vector Store)**: For semantic search, similarity matching, and content retrieval
2. **Neo4j (Graph DB)**: For structural relationships, AST/ASG storage, and graph queries

## The USS Framework
USS (Universal Semantic Structure) unifies code analysis with semantic understanding:
- **UniversalNode**: Represents any code element (function, class, variable, comment, etc.)
- **SemanticEdge**: Relationships between nodes (calls, contains, imports, similar_to, etc.)
- **UniversalGraph**: The complete knowledge graph of a codebase

## Available Operations
When users ask questions, you can perform:

### Vector Store (ChromaDB):
- `search(query, n_results, filters)` - Semantic similarity search
- `add_nodes(nodes)` - Index new content for search
- `delete(ids)` - Remove content from index

### Graph DB (Neo4j):
- `query(cypher)` - Execute Cypher queries for complex graph operations
- Common node labels: SourceFile, AST, ASG, ASTNode, ASGNode, Function, CodeAnalysis
- Common relationships: HAS_AST, HAS_ASG, HAS_ANALYSIS, CONTAINS, HAS_CHILD, EDGE, HAS_FUNCTION

## Response Format
When asked a question:
1. Determine which database(s) to query
2. Formulate the appropriate query/operation
3. Return a structured response with:
   - `action`: What you're doing
   - `database`: Which DB (neo4j, chromadb, or both)
   - `query`: The actual query/operation
   - `explanation`: Why this approach
   - `result`: The data (filled by system)

## Example Queries You Handle:
- "Find all functions with more than 5 parameters" → Neo4j Cypher query
- "Find code similar to error handling" → ChromaDB semantic search
- "What files import this module?" → Neo4j relationship traversal
- "Summarize the codebase structure" → Combine both sources
- "Index this new file" → Store in both databases

Always explain your reasoning and be precise with queries.
"""


class USSAgent:
    """Intelligent agent for managing the USS system."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize the USS Agent with database connections."""
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or os.environ.get(
            "OPENROUTER_CHAT_MODEL", "anthropic/claude-3-haiku"
        )
        self.base_url = "https://openrouter.ai/api/v1"

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var."
            )

        # Initialize database connections
        self.neo4j: Optional["Neo4jClient"] = None
        self.vector_store: Optional["VectorStore"] = None

        if NEO4J_AVAILABLE and _get_neo4j_client is not None:
            self.neo4j = _get_neo4j_client()

        if CHROMADB_AVAILABLE and _get_vector_store is not None:
            self.vector_store = _get_vector_store()

    def get_status(self) -> Dict[str, Any]:
        """Get the status of all USS components."""
        return {
            "neo4j_available": NEO4J_AVAILABLE,
            "neo4j_connected": self.neo4j.is_connected() if self.neo4j else False,
            "chromadb_available": CHROMADB_AVAILABLE,
            "model": self.model,
        }

    async def _call_llm(
        self, messages: List[Dict[str, str]], max_tokens: int = 1000
    ) -> str:
        """Make an LLM API call."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])

    async def query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a natural language query and execute appropriate operations.

        Args:
            user_query: Natural language question or command

        Returns:
            Structured response with action, query, and results
        """
        # Build context about current state
        context = f"""
Current USS Status:
- Neo4j: {'Connected' if self.neo4j and self.neo4j.is_connected() else 'Not available'}
- ChromaDB: {'Available' if self.vector_store else 'Not available'}

User Query: {user_query}

Based on the query, determine the best action and formulate the appropriate database query.
Respond in this JSON format:
{{
    "action": "description of what to do",
    "database": "neo4j" | "chromadb" | "both" | "none",
    "neo4j_query": "Cypher query if applicable",
    "chromadb_query": "search query if applicable",
    "explanation": "why this approach"
}}
"""

        messages = [
            {"role": "system", "content": USS_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        # Get LLM's decision
        llm_response = await self._call_llm(messages)

        # Parse the response (expecting JSON)
        import json

        try:
            # Try to extract JSON from the response
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                decision = json.loads(llm_response[json_start:json_end])
            else:
                decision = {"action": "parse_error", "raw": llm_response}
        except json.JSONDecodeError:
            decision = {"action": "parse_error", "raw": llm_response}

        # Execute the decided operations
        results: Dict[str, Any] = {
            "decision": decision,
            "neo4j_result": None,
            "chromadb_result": None,
        }

        # Execute Neo4j query if specified
        if decision.get("database") in ["neo4j", "both"]:
            cypher = decision.get("neo4j_query")
            if cypher and self.neo4j and self.neo4j.is_connected():
                try:
                    with self.neo4j.driver.session(database=self.neo4j.db) as session:  # type: ignore
                        result = session.run(cypher)
                        records = [dict(record) for record in result]
                        results["neo4j_result"] = {
                            "count": len(records),
                            "records": records[:20],  # Limit for response size
                        }
                except Exception as e:
                    results["neo4j_result"] = {"error": str(e)}

        # Execute ChromaDB query if specified
        if decision.get("database") in ["chromadb", "both"]:
            search_query = decision.get("chromadb_query")
            if search_query and self.vector_store:
                try:
                    search_results = self.vector_store.search(
                        search_query, n_results=10
                    )
                    results["chromadb_result"] = {
                        "count": len(search_results),
                        "results": search_results,
                    }
                except Exception as e:
                    results["chromadb_result"] = {"error": str(e)}

        # Generate final summary
        summary_prompt = f"""
Based on the query results, provide a concise answer to the user's question.

Original Question: {user_query}
Decision: {json.dumps(decision, indent=2)}
Neo4j Result: {json.dumps(results.get('neo4j_result'), indent=2) if results.get('neo4j_result') else 'None'}
ChromaDB Result: {json.dumps(results.get('chromadb_result'), indent=2) if results.get('chromadb_result') else 'None'}

Provide a clear, helpful answer.
"""

        messages = [
            {"role": "system", "content": "You are a helpful code analysis assistant."},
            {"role": "user", "content": summary_prompt},
        ]

        results["summary"] = await self._call_llm(messages, max_tokens=500)

        return results

    def query_sync(self, user_query: str) -> Dict[str, Any]:
        """Synchronous wrapper for query().

        Handles being called from both sync and async contexts (e.g., MCP server).
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()  # Just check, don't need the reference
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            return asyncio.run(self.query(user_query))

        # Already in async context - run in thread pool to avoid blocking
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, self.query(user_query))
            return future.result(timeout=120)

    async def execute_cypher(self, cypher: str) -> Dict[str, Any]:
        """Execute a raw Cypher query."""
        if not self.neo4j or not self.neo4j.is_connected():
            return {"error": "Neo4j not connected"}

        try:
            with self.neo4j.driver.session(database=self.neo4j.db) as session:  # type: ignore
                result = session.run(cypher)
                records = [dict(record) for record in result]
                return {"count": len(records), "records": records}
        except Exception as e:
            return {"error": str(e)}

    async def semantic_search(
        self, query: str, n_results: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a semantic search in ChromaDB."""
        if not self.vector_store:
            return {"error": "ChromaDB not available"}

        try:
            results = self.vector_store.search(
                query, n_results=n_results, filters=filters
            )
            return {"count": len(results), "results": results}
        except Exception as e:
            return {"error": str(e)}


# Module-level singleton
_uss_agent: Optional[USSAgent] = None


def get_uss_agent() -> USSAgent:
    """Get or create the USS Agent singleton."""
    global _uss_agent
    if _uss_agent is None:
        _uss_agent = USSAgent()
    return _uss_agent

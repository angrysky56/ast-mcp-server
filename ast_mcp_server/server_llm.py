"""
Server LLM module for Universal Semantic Structure.

Provides server-side LLM capabilities via OpenRouter to offload
processing from client AI and return lightweight responses.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional

import httpx

from ast_mcp_server.uss_core import UniversalGraph, UniversalNode

# Load .env from project root
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system env vars


class ServerLLM:
    """Server-side LLM for summarization and query answering."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or os.environ.get(
            "OPENROUTER_CHAT_MODEL", "anthropic/claude-3-haiku"
        )
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key."
            )

    async def chat(self, prompt: str, max_tokens: int = 500) -> str:
        """Send a chat completion request.

        Args:
            prompt: User prompt
            max_tokens: Maximum response tokens

        Returns:
            LLM response text
        """
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])

    async def summarize_graph(self, graph: UniversalGraph) -> str:
        """Create a lightweight summary of a graph.

        Args:
            graph: UniversalGraph to summarize

        Returns:
            Natural language summary
        """
        summary = graph.get_summary()

        # Get sample content
        samples = []
        for node in list(graph.nodes.values())[:5]:
            samples.append(f"- {node.type}: {node.preview(80)}")

        prompt = f"""Summarize this code/content structure in 2-3 sentences:

Stats:
- {summary['node_count']} nodes
- {summary['edge_count']} edges
- Node types: {summary['node_types']}
- Edge types: {summary['edge_types']}

Sample content:
{chr(10).join(samples)}

Be concise and highlight the most important aspects."""

        return await self.chat(prompt)

    async def explain_node(
        self, node: UniversalNode, context: List[UniversalNode]
    ) -> str:
        """Explain what a node does using related context.

        Args:
            node: Node to explain
            context: Related nodes for context

        Returns:
            Natural language explanation
        """
        context_text = "\n".join(
            [f"- [{c.type}] {c.preview(100)}" for c in context[:10]]
        )

        prompt = f"""Explain what this {node.type} does:

Content:
{node.content[:500]}

Related context:
{context_text}

Provide a clear, concise explanation."""

        return await self.chat(prompt)

    async def answer_query(
        self,
        query: str,
        context: List[UniversalNode],
    ) -> str:
        """Answer a natural language query using graph context.

        Args:
            query: User's question
            context: Relevant nodes from semantic search

        Returns:
            Answer based on the context
        """
        context_text = "\n\n".join(
            [f"[{c.type}] {c.content[:300]}" for c in context[:10]]
        )

        prompt = f"""Based on this code/content context, answer the question.

Context:
{context_text}

Question: {query}

Answer concisely based only on the provided context."""

        return await self.chat(prompt)

    def chat_sync(self, prompt: str, max_tokens: int = 500) -> str:
        """Synchronous wrapper for chat()."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If we are in a running loop, we can't use run_until_complete.
            # However, for an MCP tool, we should ideally be calling 'await chat()'
            # but since analyze_project is synchronous (mcp.tool decorator on non-async def),
            # we need a way to run this.
            # Using a separate thread is the safest way to run async code from sync code
            # when the event loop is already running.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self.chat(prompt, max_tokens))
                )
                return future.result()
        else:
            return loop.run_until_complete(self.chat(prompt, max_tokens))

    def summarize_graph_sync(self, graph: UniversalGraph) -> str:
        """Synchronous wrapper for summarize_graph()."""
        return asyncio.run(self.summarize_graph(graph))


# Module-level singleton
_server_llm: Optional[ServerLLM] = None


def get_server_llm() -> ServerLLM:
    """Get or create the server LLM singleton."""
    global _server_llm
    if _server_llm is None:
        _server_llm = ServerLLM()
    return _server_llm

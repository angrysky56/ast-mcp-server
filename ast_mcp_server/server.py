#!/usr/bin/env python
"""
AST/ASG Code Analysis MCP Server

Provides code structure and semantic analysis through MCP.
Includes enhanced features for scope handling, incremental parsing, and Neo4j integration.
"""

import json
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, cast

from mcp.server.fastmcp import FastMCP

from ast_mcp_server.resources import (
    CACHE_DIR,
    register_resources,
)
from ast_mcp_server.tools import register_tools

# Conditionally import optional tool modules
register_enhanced_tools: Optional[Callable[[Any], None]] = None
try:
    from ast_mcp_server.enhanced_tools import (
        register_enhanced_tools as _register_enhanced_tools,
    )

    register_enhanced_tools = _register_enhanced_tools
    ENHANCED_TOOLS_AVAILABLE = True
except ImportError:
    ENHANCED_TOOLS_AVAILABLE = False

register_transformation_tools: Optional[Callable[[Any], None]] = None
try:
    from ast_mcp_server.transformation_tools import (
        register_transformation_tools as _register_transformation_tools,
    )

    register_transformation_tools = _register_transformation_tools
    TRANSFORMATION_TOOLS_AVAILABLE = True
except ImportError:
    TRANSFORMATION_TOOLS_AVAILABLE = False

register_neo4j_tools: Optional[Callable[[Any], None]] = None
try:
    from ast_mcp_server.neo4j_tools import register_neo4j_tools as _register_neo4j_tools

    register_neo4j_tools = _register_neo4j_tools
    NEO4J_TOOLS_AVAILABLE = True
except ImportError:
    NEO4J_TOOLS_AVAILABLE = False

register_uss_agent_tools: Optional[Callable[[Any], None]] = None
try:
    from ast_mcp_server.uss_agent_tools import (
        register_uss_agent_tools as _register_uss_agent_tools,
    )

    register_uss_agent_tools = _register_uss_agent_tools
    USS_AGENT_AVAILABLE = True
except ImportError:
    USS_AGENT_AVAILABLE = False

# Initialize MCP server
mcp = FastMCP("AstAnalyzer")

# Register all tool modules
register_tools(mcp)

if ENHANCED_TOOLS_AVAILABLE and register_enhanced_tools is not None:
    register_enhanced_tools(mcp)

if TRANSFORMATION_TOOLS_AVAILABLE and register_transformation_tools is not None:
    register_transformation_tools(mcp)

if NEO4J_TOOLS_AVAILABLE and register_neo4j_tools is not None:
    register_neo4j_tools(mcp)

if USS_AGENT_AVAILABLE and register_uss_agent_tools is not None:
    register_uss_agent_tools(mcp)

register_resources(mcp)

# LRU cache for incremental parsing
MAX_AST_CACHE_SIZE = int(os.environ.get("AST_CACHE_SIZE", "100"))


class LRUCache(OrderedDict[str, Dict[str, Any]]):
    """LRU cache with maximum size limit."""

    def __init__(self, max_size: int = 100):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            oldest = next(iter(self))
            del self[oldest]

    def get_with_touch(self, key: str) -> Optional[Dict[str, Any]]:
        """Get item and move to end (most recently used)."""
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None


AST_CACHE: LRUCache = LRUCache(MAX_AST_CACHE_SIZE)


# ============================================================================
# analyze_project: Unique tool - saves analysis to files with optional AI summary
# ============================================================================
@mcp.tool()
def analyze_project(
    code: str,
    project_name: str,
    language: Optional[str] = None,
    filename: Optional[str] = None,
    include_summary: bool = True,
) -> Dict[str, Any]:
    """Save full analysis to files (functions.json, classes.json, etc). Returns file paths, not data."""
    from ast_mcp_server.output_manager import get_output_manager
    from ast_mcp_server.tools import (
        analyze_code_structure,
        create_asg_from_ast,
        parse_code_to_ast,
    )

    ast_data = parse_code_to_ast(code, language, filename)
    if "error" in ast_data:
        return ast_data

    asg_data = create_asg_from_ast(ast_data)
    analysis_data = analyze_code_structure(code, language, filename)

    output_manager = get_output_manager()
    result = output_manager.save_analysis(
        project_name=project_name,
        ast_data=ast_data,
        asg_data=asg_data,
        structure_data=analysis_data,
    )

    ai_summary = None
    if include_summary:
        try:
            from ast_mcp_server.server_llm import get_server_llm

            llm = get_server_llm()
            structure_data = analysis_data.get("structure")
            func_count = (
                len(structure_data.get("functions", [])) if structure_data else 0
            )
            class_count = (
                len(structure_data.get("classes", [])) if structure_data else 0
            )
            import_count = (
                len(structure_data.get("imports", [])) if structure_data else 0
            )

            prompt = f"""Summarize this code analysis in 2-3 sentences:

Project: {project_name}
Language: {ast_data.get("language", "unknown")}
Functions: {func_count}
Classes: {class_count}
Imports: {import_count}

Code preview (first 500 chars):
{code[:500]}

Provide a concise summary of what this code does."""

            ai_summary = llm.chat_sync(prompt, max_tokens=200)

            from pathlib import Path

            summary_path = Path(result["folder"]) / "summary.txt"
            summary_path.write_text(ai_summary)
            result["files_created"].append("summary.txt")
        except Exception as e:
            ai_summary = f"(Summary unavailable: {e})"

    response: Dict[str, Any] = {
        "status": "success",
        "project_name": project_name,
        "language": ast_data.get("language", "unknown"),
        "output_folder": result["folder"],
        "files_created": result["files_created"],
    }

    if ai_summary:
        response["summary"] = ai_summary

    return response


# ============================================================================
# Resources for cached data access
# ============================================================================
if ENHANCED_TOOLS_AVAILABLE:

    @mcp.resource("diff://{diff_hash}")
    def diff_resource(diff_hash: str) -> Dict[str, Any]:
        """Resource for cached AST diff between code versions."""
        cache_path = os.path.join(CACHE_DIR, f"{diff_hash}_diff.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    return cast(Dict[str, Any], json.load(f))
            except Exception as e:
                return {"error": f"Error reading cached diff: {e}"}
        return {"error": "Diff not found. Use diff_ast tool first."}

    @mcp.resource("enhanced_asg://{code_hash}")
    def enhanced_asg_resource(code_hash: str) -> Dict[str, Any]:
        """Resource for cached enhanced ASG."""
        from ast_mcp_server.resources import get_cache_path

        cache_path = get_cache_path(code_hash, "enhanced_asg")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    return cast(Dict[str, Any], json.load(f))
            except Exception as e:
                return {"error": f"Error reading cached enhanced ASG: {e}"}
        return {
            "error": "Enhanced ASG not found. Use generate_enhanced_asg tool first."
        }


def main() -> None:
    """Main entry point for the AST MCP Server."""
    import sys

    # CRITICAL: All startup messages must go to stderr to avoid corrupting
    # the MCP JSONRPC protocol on stdout
    print("Starting AST/ASG Code Analysis MCP Server...", file=sys.stderr)

    from ast_mcp_server.tools import init_parsers

    if not init_parsers():
        print("WARNING: Tree-sitter parsers not found. Run 'uv run build-parsers'.", file=sys.stderr)
    else:
        print("✓ Tree-sitter parsers initialized", file=sys.stderr)

    if ENHANCED_TOOLS_AVAILABLE:
        print("✓ Enhanced tools (incremental parsing, diff)", file=sys.stderr)
    if TRANSFORMATION_TOOLS_AVAILABLE:
        print("✓ Transformation tools (ast-grep)", file=sys.stderr)
    if NEO4J_TOOLS_AVAILABLE:
        print("✓ Neo4j integration", file=sys.stderr)
    if USS_AGENT_AVAILABLE:
        print("✓ USS Agent (natural language queries)", file=sys.stderr)

    print("\nRunning MCP server...", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()

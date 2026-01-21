#!/usr/bin/env python
"""
AST/ASG Code Analysis MCP Server

Provides code structure and semantic analysis through MCP.
Includes enhanced features for scope handling, incremental parsing, and Neo4j integration.
"""

import json
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, cast

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


def cached_parse_to_ast(
    code: Optional[str] = None,
    language: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse code to AST with LRU caching.

    Avoids re-parsing identical code. Cache key is MD5 hash of code + language.
    """
    import hashlib

    from ast_mcp_server.tools import detect_language, parse_code_to_ast

    # Detect language for consistent cache keys
    if not language:
        language = detect_language(code or "", filename)

    # Use filename as cache key component if code is None (lazy hash)
    # But parse requires content. tools.parse_code_to_ast handles reading,
    # so we should probably read here if we want a content hash.
    # For now, let's defer reading to the tool but use filename in hash if code is missing.
    content_key = code if code else (filename if filename else "")

    cache_key = hashlib.md5(
        f"{content_key}:{language}".encode(), usedforsecurity=False
    ).hexdigest()

    # Check cache first
    cached = AST_CACHE.get_with_touch(cache_key)
    if cached is not None:
        return cached

    # Parse and cache
    result = parse_code_to_ast(code, language, filename)
    if "error" not in result:
        AST_CACHE[cache_key] = result

    return result


# ============================================================================
# analyze_project: Unique tool - saves analysis to files with optional AI summary
# ============================================================================
@mcp.tool()
def analyze_source_file(
    project_name: str,
    code: Optional[str] = None,
    language: Optional[str] = None,
    filename: Optional[str] = None,
    include_summary: bool = True,
) -> Dict[str, Any]:
    """Analyze a single source file, save reports to disk, and optionally generate an AI summary."""
    from ast_mcp_server.output_manager import get_output_manager
    from ast_mcp_server.tools import (
        analyze_code_structure,
        create_asg_from_ast,
    )

    # Use cached parsing to avoid re-parsing identical code
    ast_data = cached_parse_to_ast(code, language, filename)
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
            # func_count, class_count, import_count replaced by detailed lists below

            # Ensure we have code for preview
            preview_code = code
            if not preview_code and filename and os.path.exists(filename):
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        preview_code = f.read()
                except Exception:
                    preview_code = ""

            # Increase limit significantly (Claude Haiku has 200k context, 15k chars is safe)
            preview_text = preview_code[:15000] if preview_code else ""

            # Extract detailed metadata
            func_names = (
                [f["name"] for f in structure_data.get("functions", [])]
                if structure_data
                else []
            )
            class_names = (
                [c["name"] for c in structure_data.get("classes", [])]
                if structure_data
                else []
            )

            # Extract relationships from ASG
            edges = asg_data.get("edges", []) if asg_data else []
            relationships = []
            for edge in edges:
                e_type = edge.get("type")
                if e_type in ["calls", "imports", "inherits", "calls_import"]:
                    relationships.append(str(e_type))

            # Count relationship types
            rel_counts: dict[str, int] = {}
            for r in relationships:
                rel_counts[r] = rel_counts.get(r, 0) + 1

            rel_summary = ", ".join([f"{k}: {v}" for k, v in rel_counts.items()])

            prompt = f"""Summarize this code file and its role in the project.

Metadata:
- Project: {project_name}
- File: {filename if filename else 'unknown'}
- Language: {ast_data.get("language", "unknown")}
- Classes: {", ".join(class_names) if class_names else "None"}
- Functions: {", ".join(func_names[:50])} {'...' if len(func_names)>50 else ''}
- Relationships detected: {rel_summary if rel_summary else "None detected"}

Code Content:
{preview_text}

Instructions:
1. Describe the main responsibility of this code.
2. List the key components (classes/functions) and what they do.
3. specific connections to other scripts or libraries based on imports and logic detected.
4. Provide a "Graph Insight" section describing how this file connects to the rest of the system.
"""

            ai_summary = llm.chat_sync(prompt, max_tokens=1000)

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


@mcp.tool()
def analyze_project(
    project_path: str,
    project_name: str,
    file_extensions: Optional[List[str]] = None,
    sync_to_db: bool = True,
    include_summary: bool = True,
) -> Dict[str, Any]:
    """Recursively analyze a project, generate reports, and optionaly sync to Graph DB.

    Args:
        project_path: Root directory to analyze
        project_name: Name of the project (for output grouping)
        file_extensions: List of extensions to include (default: .py, .js, .ts, .tsx, .go)
        sync_to_db: Whether to sync nodes/edges to Neo4j (default: True)
        include_summary: Whether to generate AI summaries for each file (default: True)
    """
    import os

    # We call the functions directly.
    # Note: analyze_source_file is defined in this module, so we can call it.
    # sync_file_to_graph is in neo4j_tools.
    from ast_mcp_server.neo4j_tools import sync_file_to_graph

    if file_extensions is None:
        file_extensions = [".py", ".js", ".ts", ".tsx", ".go"]

    processed_count = 0
    failed_count = 0
    synced_count = 0
    failures = []

    if not os.path.exists(project_path):
        return {"error": f"Project path {project_path} does not exist"}

    # Walk directory
    for root, dirs, files in os.walk(project_path):
        # Skip ignores
        dirs[:] = [
            d
            for d in dirs
            if d
            not in [
                ".git",
                "node_modules",
                "venv",
                "__pycache__",
                ".ipynb_checkpoints",
                "analyzed_projects",
            ]
        ]

        for file in files:
            ext = os.path.splitext(file)[1]
            if ext not in file_extensions:
                continue

            full_path = os.path.join(root, file)
            try:
                # Read code once
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    code_content = f.read()

                # 1. Analyze (Local JSON + Summary)
                analyze_source_file(
                    project_name=project_name,
                    code=code_content,
                    filename=full_path,
                    include_summary=include_summary,
                )
                processed_count += 1

                # 2. Sync to DB
                if sync_to_db:
                    sync_res = sync_file_to_graph(
                        code=code_content, file_path=full_path
                    )
                    if "error" not in sync_res:
                        synced_count += 1
                    else:
                        # Log sync error but don't count as full failure if analysis worked?
                        pass

            except Exception as e:
                failed_count += 1
                failures.append({"file": full_path, "error": str(e)})

    return {
        "processed_files": processed_count,
        "failed_files": failed_count,
        "synced_files": synced_count,
        "failures": failures,
    }


def main() -> None:
    """Main entry point for the AST MCP Server."""
    import sys

    # CRITICAL: All startup messages must go to stderr to avoid corrupting
    # the MCP JSONRPC protocol on stdout
    print("Starting AST/ASG Code Analysis MCP Server...", file=sys.stderr)

    from ast_mcp_server.tools import init_parsers

    if not init_parsers():
        print(
            "WARNING: Tree-sitter parsers not found. Run 'uv run build-parsers'.",
            file=sys.stderr,
        )
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

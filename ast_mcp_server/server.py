#!/usr/bin/env python
"""
AST/ASG Code Analysis MCP Server

This server provides code structure and semantic analysis capabilities through
the Model Context Protocol (MCP), allowing AI assistants to better understand
and reason about code. It includes enhanced features for improved scope handling,
incremental parsing, and performance optimizations for large codebases.
"""

import json
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, cast

from mcp.server.fastmcp import FastMCP

from ast_mcp_server.resources import (
    CACHE_DIR,
    cache_resource,
    get_code_hash,
    register_resources,
)

# Import our tools and resources
from ast_mcp_server.tools import register_tools

# Import our enhanced tools if they exist
register_enhanced_tools: Optional[Callable[[Any], None]] = None
try:
    from ast_mcp_server.enhanced_tools import (
        register_enhanced_tools as _register_enhanced_tools,
    )

    register_enhanced_tools = _register_enhanced_tools
    ENHANCED_TOOLS_AVAILABLE = True
except ImportError:
    ENHANCED_TOOLS_AVAILABLE = False

# Import our transformation tools
register_transformation_tools: Optional[Callable[[Any], None]] = None
try:
    from ast_mcp_server.transformation_tools import (
        register_transformation_tools as _register_transformation_tools,
    )

    register_transformation_tools = _register_transformation_tools
    TRANSFORMATION_TOOLS_AVAILABLE = True
except ImportError:
    TRANSFORMATION_TOOLS_AVAILABLE = False

# Import USS (Universal Semantic Structure) tools
register_uss_tools: Optional[Callable[[Any], None]] = None
try:
    from ast_mcp_server.uss_tools import register_uss_tools as _register_uss_tools

    register_uss_tools = _register_uss_tools
    USS_TOOLS_AVAILABLE = True
except ImportError:
    USS_TOOLS_AVAILABLE = False

# Import Neo4j tools
register_neo4j_tools: Optional[Callable[[Any], None]] = None
try:
    from ast_mcp_server.neo4j_tools import register_neo4j_tools as _register_neo4j_tools

    register_neo4j_tools = _register_neo4j_tools
    NEO4J_TOOLS_AVAILABLE = True
except ImportError:
    NEO4J_TOOLS_AVAILABLE = False

# Initialize the MCP server
# Initialize the MCP server
# FastMCP does not support 'version' and 'description' in __init__ in recent versions
mcp = FastMCP("AstAnalyzer")

# Register tools with the server
register_tools(mcp)

# Register enhanced tools if available
if ENHANCED_TOOLS_AVAILABLE and register_enhanced_tools is not None:
    register_enhanced_tools(mcp)

# Register transformation tools if available
if TRANSFORMATION_TOOLS_AVAILABLE and register_transformation_tools is not None:
    register_transformation_tools(mcp)

# Register USS tools if available
if USS_TOOLS_AVAILABLE and register_uss_tools is not None:
    register_uss_tools(mcp)

# Register Neo4j tools if available
if NEO4J_TOOLS_AVAILABLE and register_neo4j_tools is not None:
    register_neo4j_tools(mcp)

# Import USS Agent tools
register_uss_agent_tools: Optional[Callable[[Any], None]] = None
try:
    from ast_mcp_server.uss_agent_tools import (
        register_uss_agent_tools as _register_uss_agent_tools,
    )

    register_uss_agent_tools = _register_uss_agent_tools
    USS_AGENT_AVAILABLE = True
except ImportError:
    USS_AGENT_AVAILABLE = False

# Register USS Agent tools if available
if USS_AGENT_AVAILABLE and register_uss_agent_tools is not None:
    register_uss_agent_tools(mcp)

# Register resources with the server
register_resources(mcp)

# LRU cache for storing previous ASTs for incremental parsing
MAX_AST_CACHE_SIZE = int(os.environ.get("AST_CACHE_SIZE", "100"))


class LRUCache(OrderedDict[str, Dict[str, Any]]):
    """LRU cache with maximum size limit to prevent memory leaks."""

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
        """Get an item and move it to the end (most recently used)."""
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None


AST_CACHE: LRUCache = LRUCache(MAX_AST_CACHE_SIZE)

# Add custom handlers for tool operations
# These ensure that results are cached for resource access


@mcp.tool()
def parse_and_cache(
    code: str, language: Optional[str] = None, filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse code into an AST and cache it for resource access.

    This tool parses source code into an Abstract Syntax Tree and stores it
    for later retrieval as a resource. It returns both the AST data and
    a resource URI that can be used to access the data.

    Args:
        code: Source code to parse
        language: Programming language (optional, will be auto-detected if not provided)
        filename: Source filename (optional, helps with language detection)

    Returns:
        Dictionary with AST data and resource URI
    """
    from ast_mcp_server.tools import parse_code_to_ast

    # Generate a hash for the code
    code_hash = get_code_hash(code)

    # Parse the code to AST
    ast_data = parse_code_to_ast(code, language, filename)

    # Cache the result
    if "error" not in ast_data:
        cache_resource(code, "ast", ast_data)

        # Return the AST with a resource URI
        return {"ast": ast_data, "resource_uri": f"ast://{code_hash}"}
    else:
        return ast_data


@mcp.tool()
def generate_and_cache_asg(
    code: str, language: Optional[str] = None, filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate an ASG from code and cache it for resource access.

    This tool analyzes source code to create an Abstract Semantic Graph and
    stores it for later retrieval as a resource. It returns both the ASG data
    and a resource URI that can be used to access the data.

    Args:
        code: Source code to analyze
        language: Programming language (optional, will be auto-detected if not provided)
        filename: Source filename (optional, helps with language detection)

    Returns:
        Dictionary with ASG data and resource URI
    """
    from ast_mcp_server.tools import create_asg_from_ast, parse_code_to_ast

    # Generate a hash for the code
    code_hash = get_code_hash(code)

    # Parse to AST first
    ast_data = parse_code_to_ast(code, language, filename)

    if "error" in ast_data:
        return ast_data

    # Generate ASG
    asg_data = create_asg_from_ast(ast_data)

    # Cache both results
    cache_resource(code, "ast", ast_data)
    cache_resource(code, "asg", asg_data)

    # Return the ASG with a resource URI
    return {"asg": asg_data, "resource_uri": f"asg://{code_hash}"}


@mcp.tool()
def analyze_and_cache(
    code: str, language: Optional[str] = None, filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze code structure and cache the results for resource access.

    This tool analyzes source code structure and stores the results
    for later retrieval as a resource. It returns both the analysis data
    and a resource URI that can be used to access the data.

    Args:
        code: Source code to analyze
        language: Programming language (optional, will be auto-detected if not provided)
        filename: Source filename (optional, helps with language detection)

    Returns:
        Dictionary with analysis data and resource URI
    """
    from ast_mcp_server.tools import analyze_code_structure

    # Generate a hash for the code
    code_hash = get_code_hash(code)

    # Analyze the code
    analysis_data = analyze_code_structure(code, language, filename)

    # Cache the result
    if "error" not in analysis_data:
        cache_resource(code, "analysis", analysis_data)

        # Return the analysis with a resource URI
        return {"analysis": analysis_data, "resource_uri": f"analysis://{code_hash}"}
    else:
        return analysis_data


@mcp.tool()
def analyze_project(
    code: str,
    project_name: str,
    language: Optional[str] = None,
    filename: Optional[str] = None,
    include_summary: bool = True,
) -> Dict[str, Any]:
    """
    Analyze code and save results to analyzed_projects folder.

    Instead of returning vast AST data, this tool saves analysis to
    organized files and returns file paths. The output is structured
    into logical sections:
    - functions.json - All function definitions
    - classes.json - All class definitions
    - imports.json - All imports/dependencies
    - structure.json - Code metrics and overview
    - summary.txt - AI-generated summary (if include_summary=True and LLM configured)

    Args:
        code: Source code to analyze
        project_name: Name for the project (used in output folder name)
        language: Programming language (optional, auto-detected)
        filename: Source filename (optional)
        include_summary: Generate AI summary using server LLM (default True)

    Returns:
        Dictionary with file paths to saved analysis, NOT the full AST
    """
    from ast_mcp_server.output_manager import get_output_manager
    from ast_mcp_server.tools import (
        analyze_code_structure,
        create_asg_from_ast,
        parse_code_to_ast,
    )

    # Parse and analyze
    ast_data = parse_code_to_ast(code, language, filename)
    if "error" in ast_data:
        return ast_data

    asg_data = create_asg_from_ast(ast_data)
    structure_data = analyze_code_structure(code, language, filename)

    # Save to organized files
    output_manager = get_output_manager()
    result = output_manager.save_analysis(
        project_name=project_name,
        ast_data=ast_data,
        asg_data=asg_data if "error" not in asg_data else None,
        structure_data=structure_data if "error" not in structure_data else None,
        code=code,
    )

    # Generate AI summary if requested and LLM is available
    ai_summary = None
    if include_summary:
        try:
            from ast_mcp_server.server_llm import get_server_llm

            llm = get_server_llm()

            # Build context for summary
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

            # Save summary to file
            from pathlib import Path

            summary_path = Path(result["folder"]) / "summary.txt"
            summary_path.write_text(ai_summary)
            result["files_created"].append("summary.txt")

        except Exception as e:
            ai_summary = f"(Summary unavailable: {e})"

    # Return paths, not the vast data
    response: Dict[str, Any] = {
        "status": "success",
        "project_name": project_name,
        "language": ast_data.get("language", "unknown"),
        "output_folder": result["folder"],
        "files_created": result["files_created"],
        "message": f"Analysis saved to {result['folder']}",
        "tip": "View individual files: functions.json, classes.json, imports.json, structure.json",
    }

    if ai_summary:
        response["summary"] = ai_summary

    return response


# Enhanced tools from server_enhanced.py
if ENHANCED_TOOLS_AVAILABLE:
    from ast_mcp_server.enhanced_tools import (
        create_enhanced_asg_from_ast,
        generate_ast_diff,
        parse_code_to_ast_incremental,
    )

    @mcp.tool()
    def parse_and_cache_incremental(
        code: str,
        language: Optional[str] = None,
        filename: Optional[str] = None,
        code_id: Optional[
            str
        ] = None,  # Optional identifier for the code (e.g. file path)
    ) -> Dict[str, Any]:
        """
        Parse code into an AST incrementally and cache it for resource access.

        This tool uses incremental parsing when possible, which is much faster
        for large files with small changes. It also caches the results for
        future access.

        Args:
            code: Source code to parse
            language: Programming language (optional, will be auto-detected if not provided)
            filename: Source filename (optional, helps with language detection)
            code_id: Optional identifier for the code (e.g. file path)

        Returns:
            Dictionary with AST data and resource URI
        """
        # from ast_mcp_server.enhanced_tools import parse_code_to_ast_incremental # Moved to top of block

        # Generate a hash for the code
        code_hash = get_code_hash(code)

        # Use file path as cache key if provided, otherwise use hash
        cache_key = code_id if code_id else code_hash

        # Check if we have a previous version in cache
        old_code = None
        if cache_key in AST_CACHE:
            old_code = AST_CACHE["code"]

        # Parse the code to AST, potentially using incremental parsing
        ast_data = parse_code_to_ast_incremental(code, language, filename)

        # Cache the current code for future incremental parsing
        AST_CACHE[cache_key] = {
            "code": code,
            "ast_data": ast_data,
            "language": ast_data.get("language"),
        }

        # Cache the result for resource access
        if "error" not in ast_data:
            cache_resource(code, "ast", ast_data)

            # Return the AST with a resource URI
            return {
                "ast": ast_data,
                "resource_uri": f"ast://{code_hash}",
                "incremental": old_code is not None,
            }
        else:
            return ast_data

    @mcp.tool()
    def generate_and_cache_enhanced_asg(
        code: str, language: Optional[str] = None, filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an enhanced ASG from code and cache it for resource access.

        This tool creates a more complete semantic graph with better
        scope handling, control flow edges, and data flow edges. It stores
        the results for later retrieval.

        Args:
            code: Source code to analyze
            language: Programming language (optional, will be auto-detected if not provided)
            filename: Source filename (optional, helps with language detection)

        Returns:
            Dictionary with enhanced ASG data and resource URI
        """
        # from ast_mcp_server.enhanced_tools import parse_code_to_ast_incremental, create_enhanced_asg_from_ast # Moved to top of block

        # Generate a hash for the code
        code_hash = get_code_hash(code)

        # Parse to AST first
        ast_data = parse_code_to_ast_incremental(code, language, filename)

        if "error" in ast_data:
            return ast_data

        # Generate enhanced ASG
        asg_data = create_enhanced_asg_from_ast(ast_data)

        # Cache both results
        cache_resource(code, "ast", ast_data)
        cache_resource(code, "enhanced_asg", asg_data)

        # Return the ASG with a resource URI
        return {"asg": asg_data, "resource_uri": f"enhanced_asg://{code_hash}"}

    @mcp.tool()
    def ast_diff_and_cache(
        old_code: str,
        new_code: str,
        language: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an AST diff between old and new code versions and cache it.

        This tool compares two versions of code and returns only the changed AST nodes,
        which is much more efficient for large files with small changes.

        Args:
            old_code: Previous version of the code
            new_code: New version of the code
            language: Programming language (optional, will be auto-detected if not provided)
            filename: Source filename (optional, helps with language detection)

        Returns:
            Dictionary with diff data and resource URIs
        """
        # Parse old and new code to ASTs
        ast_old = parse_code_to_ast_incremental(
            old_code, language=language, filename=filename
        )
        ast_new = parse_code_to_ast_incremental(
            new_code, language=language, filename=filename
        )

        if "error" in ast_old:
            return ast_old
        if "error" in ast_new:
            return ast_new

        # Generate the diff using the imported generate_ast_diff function
        diff_data = generate_ast_diff(ast_old, ast_new, old_code, new_code)

        if "error" in diff_data:
            return diff_data

        # Generate hashes for both code versions
        old_hash = get_code_hash(old_code)
        new_hash = get_code_hash(new_code)

        # Cache the diff
        diff_hash = get_code_hash(f"{old_hash}_{new_hash}")
        cache_resource(f"{old_hash}_{new_hash}", "diff", diff_data)

        # Return the diff with a resource URI
        return {
            "diff": diff_data,
            "resource_uri": f"diff://{diff_hash}",
            "old_uri": f"ast://{old_hash}",
            "new_uri": f"ast://{new_hash}",
        }

    # Register enhanced resources
    @mcp.resource("diff://{diff_hash}")
    def diff_resource(diff_hash: str) -> Dict[str, Any]:
        """
        Resource that provides an AST diff between two code versions.

        The diff_hash is derived from the hashes of the old and new code versions.

        Args:
            diff_hash: Hash of the diff to retrieve

        Returns:
            The cached diff data
        """
        cache_path = os.path.join(CACHE_DIR, f"{diff_hash}_diff.json")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    return cast(Dict[str, Any], json.load(f))
            except Exception as e:
                return {"error": f"Error reading cached diff: {e}"}

        return {"error": "Diff not found. Please use ast_diff_and_cache tool first."}

    @mcp.resource("enhanced_asg://{code_hash}")
    def enhanced_asg_resource(code_hash: str) -> Dict[str, Any]:
        """
        Resource that provides the enhanced Abstract Semantic Graph for a piece of code.

        The code_hash is used to locate the cached enhanced ASG.

        Args:
            code_hash: Hash of the code to retrieve enhanced ASG for

        Returns:
            The cached enhanced ASG data
        """
        from ast_mcp_server.resources import get_cache_path

        cache_path = get_cache_path(code_hash, "enhanced_asg")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    return cast(Dict[str, Any], json.load(f))
            except Exception as e:
                return {"error": f"Error reading cached enhanced ASG: {e}"}

        return {
            "error": "Enhanced ASG not found. Please use generate_and_cache_enhanced_asg tool first."
        }


def main() -> None:
    """Main entry point for the AST MCP Server."""
    print("Starting server initialization...")

    # Check if tree-sitter parsers are available
    from ast_mcp_server.tools import init_parsers

    print("Checking for tree-sitter parsers...")
    if not init_parsers():
        print("WARNING: Tree-sitter language parsers not found.")
        print("Run 'uv run build-parsers' to build the parsers.")
        print("Some functionality may be limited.")
    else:
        print("Tree-sitter parsers initialized successfully!")

    # Report on enhanced tools availability
    if ENHANCED_TOOLS_AVAILABLE:
        print("Enhanced AST/ASG analysis tools are available.")
    else:
        print("Enhanced tools module not found. Only basic functionality is available.")
        print("Create ast_mcp_server/enhanced_tools.py to enable advanced features.")

    # Report on transformation tools availability
    if TRANSFORMATION_TOOLS_AVAILABLE:
        print("Code transformation tools (ast-grep integration) are available.")
    else:
        print(
            "Transformation tools not available. Install ast-grep-cli for code transformation features."
        )

    # Report on USS (Universal Semantic Structure) tools availability
    if USS_TOOLS_AVAILABLE:
        print("USS (Universal Semantic Structure) tools are available.")
        print("  - Semantic search via ChromaDB")
        print("  - Graph traversal and queries")
        print("  - Set OPENROUTER_API_KEY for embeddings and server LLM")
    else:
        print("USS tools not available.")

    # Report on Neo4j tools availability
    if NEO4J_TOOLS_AVAILABLE:
        print("Neo4j integration tools are available.")
        print("  - Sync code analysis to Graph DB")
        print("  - Cypher query execution")
    else:
        print("Neo4j tools not available. Install neo4j package.")

    # Report on USS Agent availability
    if USS_AGENT_AVAILABLE:
        print("USS Agent is available.")
        print("  - Natural language queries to both databases")
        print("  - Intelligent curation of code knowledge")

    # Start the MCP server
    print("Starting AST/ASG Code Analysis MCP Server v0.3.0...")
    print("Running MCP server...")
    mcp.run()
    print("MCP server exited.")  # This will only print if mcp.run() returns


if __name__ == "__main__":
    main()

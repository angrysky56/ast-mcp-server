#!/usr/bin/env python
"""
AST/ASG Code Analysis MCP Server

This server provides code structure and semantic analysis capabilities through
the Model Context Protocol (MCP), allowing AI assistants to better understand
and reason about code. It includes enhanced features for improved scope handling,
incremental parsing, and performance optimizations for large codebases.
"""

import os
import json
from mcp.server.fastmcp import FastMCP
from typing import Dict, Optional

# Import our tools and resources
from ast_mcp_server.tools import register_tools
from ast_mcp_server.resources import register_resources, cache_resource, get_code_hash, CACHE_DIR

# Import our enhanced tools if they exist
try:
    from ast_mcp_server.enhanced_tools import register_enhanced_tools
    ENHANCED_TOOLS_AVAILABLE = True
except ImportError:
    register_enhanced_tools = None
    ENHANCED_TOOLS_AVAILABLE = False

# Initialize the MCP server
mcp = FastMCP(
    "AstAnalyzer",
    version="0.2.0",
    description="Code structure and semantic analysis using AST/ASG with enhanced features"
)

# Register tools with the server
register_tools(mcp)

# Register enhanced tools if available
if ENHANCED_TOOLS_AVAILABLE and register_enhanced_tools is not None:
    register_enhanced_tools(mcp)

# Register resources with the server
register_resources(mcp)

# Cache for storing previous ASTs for incremental parsing
AST_CACHE = {}

# Add custom handlers for tool operations
# These ensure that results are cached for resource access

@mcp.tool()
def parse_and_cache(code: str, language: Optional[str] = None, filename: Optional[str] = None) -> Dict:
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
        return {
            "ast": ast_data,
            "resource_uri": f"ast://{code_hash}"
        }
    else:
        return ast_data

@mcp.tool()
def generate_and_cache_asg(code: str, language: Optional[str] = None, filename: Optional[str] = None) -> Dict:
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
    from ast_mcp_server.tools import parse_code_to_ast, create_asg_from_ast

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
    return {
        "asg": asg_data,
        "resource_uri": f"asg://{code_hash}"
    }

@mcp.tool()
def analyze_and_cache(code: str, language: Optional[str] = None, filename: Optional[str] = None) -> Dict:
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
        return {
            "analysis": analysis_data,
            "resource_uri": f"analysis://{code_hash}"
        }
    else:
        return analysis_data

# Enhanced tools from server_enhanced.py
if ENHANCED_TOOLS_AVAILABLE:
    from ast_mcp_server.enhanced_tools import parse_code_to_ast_incremental, create_enhanced_asg_from_ast, generate_ast_diff

    @mcp.tool()
    def parse_and_cache_incremental(
        code: str,
        language: Optional[str] = None,
        filename: Optional[str] = None,
        code_id: Optional[str] = None  # Optional identifier for the code (e.g. file path)
    ) -> Dict:
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
            "language": ast_data.get("language")
        }

        # Cache the result for resource access
        if "error" not in ast_data:
            cache_resource(code, "ast", ast_data)

            # Return the AST with a resource URI
            return {
                "ast": ast_data,
                "resource_uri": f"ast://{code_hash}",
                "incremental": old_code is not None
            }
        else:
            return ast_data

    @mcp.tool()
    def generate_and_cache_enhanced_asg(
        code: str,
        language: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Dict:
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
        return {
            "asg": asg_data,
            "resource_uri": f"enhanced_asg://{code_hash}"
        }

    @mcp.tool()
    def ast_diff_and_cache(
        old_code: str,
        new_code: str,
        language: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Dict:
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
        ast_old = parse_code_to_ast_incremental(old_code, language=language, filename=filename)
        ast_new = parse_code_to_ast_incremental(new_code, language=language, filename=filename)

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
            "new_uri": f"ast://{new_hash}"
        }

    # Register enhanced resources
    @mcp.resource("diff://{diff_hash}")
    def diff_resource(diff_hash: str) -> Dict:
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
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                return {"error": f"Error reading cached diff: {e}"}

        return {"error": "Diff not found. Please use ast_diff_and_cache tool first."}

    @mcp.resource("enhanced_asg://{code_hash}")
    def enhanced_asg_resource(code_hash: str) -> Dict:
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
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                return {"error": f"Error reading cached enhanced ASG: {e}"}

        return {"error": "Enhanced ASG not found. Please use generate_and_cache_enhanced_asg tool first."}

def main():
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

    # Start the MCP server
    print("Starting AST/ASG Code Analysis MCP Server...")
    print("Running MCP server...")
    mcp.run()
    print("MCP server exited.")  # This will only print if mcp.run() returns


if __name__ == "__main__":
    main()

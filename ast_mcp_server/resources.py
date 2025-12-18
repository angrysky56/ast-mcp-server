"""
AST/ASG resources for the MCP server.

This module defines the resources that provide code structure and semantic information
through the Model Context Protocol.
"""

import hashlib
import json
import os
import sys
import tempfile
from typing import Any, Dict, Optional, cast

# from .tools import parse_code_to_ast, create_asg_from_ast, analyze_code_structure # These are not used directly in this file

# Directory to store cached ASTs and ASGs
CACHE_DIR = os.path.join(tempfile.gettempdir(), "ast_mcp_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(code_hash: str, resource_type: str) -> str:
    """Get the cache file path for a given code hash and resource type."""
    return os.path.join(CACHE_DIR, f"{code_hash}_{resource_type}.json")


def get_code_hash(code: str) -> str:
    """Generate a hash for the code to use as a cache key."""
    return hashlib.md5(code.encode("utf-8"), usedforsecurity=False).hexdigest()


def cache_resource(code: str, resource_type: str, data: Dict[str, Any]) -> None:
    """Cache a resource for faster retrieval."""
    code_hash = get_code_hash(code)
    cache_path = get_cache_path(code_hash, resource_type)

    try:
        with open(cache_path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error caching resource: {e}", file=sys.stderr)


def get_cached_resource(code: str, resource_type: str) -> Optional[Dict[str, Any]]:
    """Get a cached resource if available."""
    code_hash = get_code_hash(code)
    cache_path = get_cache_path(code_hash, resource_type)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return cast(Dict[str, Any], json.load(f))
        except Exception as e:
            print(f"Error reading cached resource: {e}", file=sys.stderr)

    return None


def register_resources(mcp_server: Any) -> None:
    """Register all resources with the MCP server."""

    @mcp_server.resource("ast://{code_hash}")
    def ast_resource(code_hash: str) -> Dict[str, Any]:
        """
        Resource that provides the Abstract Syntax Tree for a piece of code.

        The code_hash is used to locate the cached AST. If not found, this will
        return an error - the AST should be created first using the parse_to_ast tool.

        Args:
            code_hash: Hash of the code to retrieve AST for

        Returns:
            The cached AST data
        """
        cache_path = get_cache_path(code_hash, "ast")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    return cast(Dict[str, Any], json.load(f))
            except Exception as e:
                return {"error": f"Error reading cached AST: {e}"}

        return {"error": "AST not found. Please use parse_to_ast tool first."}

    @mcp_server.resource("asg://{code_hash}")
    def asg_resource(code_hash: str) -> Dict[str, Any]:
        """
        Resource that provides the Abstract Semantic Graph for a piece of code.

        The code_hash is used to locate the cached ASG. If not found, this will
        return an error - the ASG should be created first using the generate_asg tool.

        Args:
            code_hash: Hash of the code to retrieve ASG for

        Returns:
            The cached ASG data
        """
        cache_path = get_cache_path(code_hash, "asg")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    return cast(Dict[str, Any], json.load(f))
            except Exception as e:
                return {"error": f"Error reading cached ASG: {e}"}

        return {"error": "ASG not found. Please use generate_asg tool first."}

    @mcp_server.resource("analysis://{code_hash}")
    def analysis_resource(code_hash: str) -> Dict[str, Any]:
        """
        Resource that provides code structure analysis.

        The code_hash is used to locate the cached analysis. If not found, this will
        return an error - the analysis should be created first using the analyze_code tool.

        Args:
            code_hash: Hash of the code to retrieve analysis for

        Returns:
            The cached analysis data
        """
        cache_path = get_cache_path(code_hash, "analysis")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    return cast(Dict[str, Any], json.load(f))
            except Exception as e:
                return {"error": f"Error reading cached analysis: {e}"}

        return {"error": "Analysis not found. Please use analyze_code tool first."}

    # Custom resource handler for AST node detail
    @mcp_server.resource("ast://{code_hash}/node/{node_id}")
    def ast_node_resource(code_hash: str, node_id: str) -> Dict[str, Any]:
        """
        Resource that provides details about a specific AST node.

        Args:
            code_hash: Hash of the code containing the AST
            node_id: ID of the node to retrieve

        Returns:
            The node details
        """
        # Get the full AST
        cache_path = get_cache_path(code_hash, "ast")

        if not os.path.exists(cache_path):
            return {"error": "AST not found. Please use parse_to_ast tool first."}

        try:
            with open(cache_path, "r") as f:
                ast_data = cast(Dict[str, Any], json.load(f))

            # Find the node by its ID
            def find_node(
                node: Dict[str, Any], target_id: str
            ) -> Optional[Dict[str, Any]]:
                # Generate this node's ID
                current_id = f"{node['type']}_{node['start_byte']}_{node['end_byte']}"

                if current_id == target_id:
                    return node

                # Search in children
                for child in node.get("children", []):
                    result = find_node(child, target_id)
                    if result:
                        return result

                return None

            node = find_node(ast_data["ast"], node_id)

            if node:
                return node
            else:
                return {"error": f"Node with ID {node_id} not found in the AST"}

        except Exception as e:
            return {"error": f"Error retrieving AST node: {e}"}

"""
Enhanced AST/ASG analysis tools for the MCP server.

This module provides improved implementations of the AST/ASG analysis tools
with better scope handling, more complete edge detection, and performance
optimizations for handling large codebases.
"""

from collections import defaultdict
from typing import Dict, List, Optional

from tree_sitter import Parser, Tree

from .tools import LANGUAGE_MAP, detect_language, init_parsers, languages, node_to_dict

# Create a parser instance
parser = Parser()

# Types of control flow nodes in Python
PYTHON_CONTROL_FLOW_NODES = {
    "if_statement",
    "for_statement",
    "while_statement",
    "try_statement",
    "with_statement",
    "match_statement",
}

# Types of nodes that create new scopes in Python
PYTHON_SCOPE_NODES = {
    "function_definition",
    "class_definition",
    "for_statement",
    "while_statement",
    "with_statement",
}


class ScopeManager:
    """Manages scope hierarchy for semantic analysis."""

    def __init__(self):
        self.scopes = {}  # Maps scope_id to parent_scope_id
        self.variables = defaultdict(dict)  # Maps scope_id -> {var_name: var_id}
        self.functions = {}  # Maps func_name to func_id
        self.classes = {}  # Maps class_name to class_id
        self.imports = {}  # Maps import_name to import_id
        self.global_scope = "global"
        self.control_flow = []  # Stack of control flow nodes

    def enter_scope(self, scope_id: str, parent_scope_id: Optional[str] = None) -> str:
        """
        Enter a new scope.

        Args:
            scope_id: ID of the new scope
            parent_scope_id: ID of the parent scope (None for global scope)

        Returns:
            The scope_id for chaining
        """
        if parent_scope_id is None:
            parent_scope_id = self.global_scope

        self.scopes[scope_id] = parent_scope_id
        return scope_id

    def get_parent_scope(self, scope_id: str) -> Optional[str]:
        """Get the parent scope of the given scope."""
        return self.scopes.get(scope_id)

    def add_variable(self, var_name: str, var_id: str, scope_id: str) -> None:
        """Add a variable to the current scope."""
        self.variables[scope_id][var_name] = var_id

    def add_function(self, func_name: str, func_id: str) -> None:
        """Add a function definition."""
        self.functions[func_name] = func_id

    def add_class(self, class_name: str, class_id: str) -> None:
        """Add a class definition."""
        self.classes[class_name] = class_id

    def add_import(self, import_name: str, import_id: str) -> None:
        """Add an import definition."""
        self.imports[import_name] = import_id

    def find_variable(self, var_name: str, scope_id: str) -> Optional[str]:
        """
        Find a variable in the scope hierarchy.
        Looks in current scope first, then parent scopes.

        Args:
            var_name: Name of the variable to find
            scope_id: ID of the scope to start the search from

        Returns:
            The variable ID if found, None otherwise
        """
        current_scope = scope_id

        while current_scope is not None:
            if var_name in self.variables[current_scope]:
                return self.variables[current_scope][var_name]

            # Move up to parent scope
            current_scope = self.get_parent_scope(current_scope)

        return None

    def find_function(self, func_name: str) -> Optional[str]:
        """Find a function by name."""
        return self.functions.get(func_name)

    def find_class(self, class_name: str) -> Optional[str]:
        """Find a class by name."""
        return self.classes.get(class_name)

    def find_import(self, import_name: str) -> Optional[str]:
        """Find an import by name."""
        return self.imports.get(import_name)

    def enter_control_flow(self, node_id: str) -> None:
        """Push a control flow node onto the stack."""
        self.control_flow.append(node_id)

    def exit_control_flow(self) -> Optional[str]:
        """Pop a control flow node from the stack."""
        if self.control_flow:
            return self.control_flow.pop()
        return None

    def get_current_control_flow(self) -> Optional[str]:
        """Get the current control flow node."""
        if self.control_flow:
            return self.control_flow[-1]
        return None


def parse_code_to_ast_incremental(
    code: str,
    language: Optional[str] = None,
    filename: Optional[str] = None,
    previous_tree: Optional[Tree] = None,
    old_code: Optional[str] = None,
    include_children: bool = True,
) -> Dict:
    """
    Parse code into an AST incrementally using Tree-sitter.

    This is an optimized version that can use a previous tree to
    only parse the changed parts of the code, which is much faster
    for large files with small changes.

    Args:
        code: Source code to parse
        language: Programming language identifier (optional)
        filename: Source file name (optional, used for language detection)
        previous_tree: Previously parsed tree (optional, for incremental parsing)
        old_code: Previous version of the code (required if previous_tree is provided)
        include_children: Whether to include child nodes in the result

    Returns:
        Dictionary representation of the AST, with additional metadata
        for incremental parsing
    """
    # Initialize parsers if not done already
    if not languages and not init_parsers():
        return {
            "error": "Tree-sitter language parsers not available. Run build_parsers.py first."
        }

    # Detect language if not provided
    if not language:
        language = detect_language(code, filename)

    # Normalize language identifier
    language = LANGUAGE_MAP.get(language.lower(), language.lower())

    # Check if language is supported
    if language not in languages:
        return {"error": f"Unsupported language: {language}"}

    try:
        # Set the parser language
        parser.language = languages[language]

        # Parse the code, potentially incrementally
        source_bytes = bytes(code, "utf-8")

        if previous_tree and old_code:
            tree = parser.parse(source_bytes, previous_tree)

            # Calculate which nodes changed
            changed_ranges = []
            for edit in tree.changed_ranges(previous_tree):
                changed_ranges.append(
                    {
                        "start_byte": edit.start_byte,
                        "end_byte": edit.end_byte,
                        "start_point": {
                            "row": edit.start_point[0],
                            "column": edit.start_point[1],
                        },
                        "end_point": {
                            "row": edit.end_point[0],
                            "column": edit.end_point[1],
                        },
                    }
                )
        else:
            tree = parser.parse(source_bytes)
            changed_ranges = None  # No previous tree to compare with

        # Convert to dictionary
        root_node = tree.root_node
        ast = node_to_dict(root_node, source_bytes, include_children)

        # NOTE: tree_object removed from result to prevent C-based Tree reference leaks
        # The tree object is only needed internally for incremental parsing
        result = {
            "language": language,
            "ast": ast,
        }

        if changed_ranges:
            result["changed_ranges"] = changed_ranges

        return result

    except Exception as e:
        return {"error": f"Error parsing code: {e}"}


def create_enhanced_asg_from_ast(ast_data: Dict) -> Dict:
    """
    Create an enhanced Abstract Semantic Graph (ASG) from an AST.

    This version provides more complete edge detection, including improved
    scope handling and control flow edges.

    Args:
        ast_data: AST data from parse_code_to_ast

    Returns:
        Dictionary representation of the enhanced ASG
    """
    if "error" in ast_data:
        return ast_data

    ast = ast_data["ast"]
    language = ast_data["language"]

    # Extract nodes and edges from the AST
    nodes = []
    edges = []
    node_ids = {}  # Map of {node_id: node_index} for quick lookups

    def extract_nodes(node, parent_id=None):
        node_id = f"{node['type']}_{node['start_byte']}_{node['end_byte']}"

        # Create a node object with metadata
        node_index = len(nodes)
        node_obj = {
            "id": node_id,
            "type": node["type"],
            "text": node["text"],
            "start_byte": node["start_byte"],
            "end_byte": node["end_byte"],
            "start_line": node["start_point"]["row"],
            "start_col": node["start_point"]["column"],
            "end_line": node["end_point"]["row"],
            "end_col": node["end_point"]["column"],
        }
        nodes.append(node_obj)
        node_ids[node_id] = node_index

        # Add edge to parent if exists
        if parent_id:
            edges.append({"source": parent_id, "target": node_id, "type": "contains"})

        # Process children
        if "children" in node:
            for child in node["children"]:
                extract_nodes(child, node_id)

        return node_id

    # Start extraction from the root
    root_id = extract_nodes(ast)

    # Add semantic edges based on language-specific rules
    if language == "python":
        add_enhanced_python_semantic_edges(ast, edges)
    elif language in ["javascript", "typescript"]:
        add_enhanced_js_ts_semantic_edges(ast, edges)

    # Add additional metadata to the ASG
    return {
        "language": language,
        "nodes": nodes,
        "edges": edges,
        "root": root_id,
        "node_lookup": node_ids,  # Helps with quick node lookup by ID
    }


def add_enhanced_python_semantic_edges(ast: Dict, edges: List[Dict]):
    """
    Add enhanced Python-specific semantic edges to the ASG.

    This version provides more complete edge detection, including:
    - Proper scope hierarchy and variable resolution
    - Control flow edges between blocks
    - Data flow edges showing variable dependencies

    Args:
        ast: The Python AST
        edges: List to store the detected edges
    """
    scope_manager = ScopeManager()
    current_scope = scope_manager.global_scope

    # First pass: find all definitions (functions, classes, variables)
    def find_enhanced_definitions(node, scope=None):
        nonlocal current_scope
        old_scope = current_scope
        node_id = f"{node['type']}_{node['start_byte']}_{node['end_byte']}"

        # Check for scope-creating nodes
        if node["type"] in PYTHON_SCOPE_NODES:
            # Create new scope for this node
            current_scope = scope_manager.enter_scope(node_id, current_scope)

        # Check for definitions
        if node["type"] == "function_definition":
            # Get function name
            for child in node.get("children", []):
                if child["type"] == "identifier":
                    func_name = child["text"]
                    func_id = node_id
                    scope_manager.add_function(func_name, func_id)

                    # Add parameters to function scope
                    for param_child in node.get("children", []):
                        if param_child["type"] == "parameters":
                            for param in param_child.get("children", []):
                                if param["type"] == "identifier":
                                    param_name = param["text"]
                                    param_id = f"{param['type']}_{param['start_byte']}_{param['end_byte']}"
                                    scope_manager.add_variable(
                                        param_name, param_id, current_scope
                                    )

                    break

        elif node["type"] == "class_definition":
            # Get class name
            for child in node.get("children", []):
                if child["type"] == "identifier":
                    class_name = child["text"]
                    class_id = node_id
                    scope_manager.add_class(class_name, class_id)
                    break

        elif node["type"] == "assignment":
            # Handle variable assignments
            # The left side is the target (variable being defined)
            targets = []

            # Split children into targets and values
            for i, child in enumerate(node.get("children", [])):
                if child["type"] == "=" and i > 0:
                    # Everything before '=' is a target
                    targets = node["children"][:i]
                    # Everything after '=' is a value (processed by generic recursion)
                    # values = node["children"][i+1:] # Removed as `values` list is unused
                    break

            # Process targets (variables being assigned)
            for target in targets:
                if target["type"] == "identifier":
                    var_name = target["text"]
                    var_id = (
                        f"{target['type']}_{target['start_byte']}_{target['end_byte']}"
                    )
                    scope_manager.add_variable(var_name, var_id, current_scope)

                # Handle tuple unpacking
                elif target["type"] == "tuple" or target["type"] == "list":
                    for element in target.get("children", []):
                        if element["type"] == "identifier":
                            var_name = element["text"]
                            var_id = f"{element['type']}_{element['start_byte']}_{element['end_byte']}"
                            scope_manager.add_variable(var_name, var_id, current_scope)

        elif (
            node["type"] == "import_statement"
            or node["type"] == "import_from_statement"
        ):
            # Track imported modules and functions
            for child in node.get("children", []):
                if child["type"] == "dotted_name" or child["type"] == "identifier":
                    import_name = child["text"]
                    import_id = (
                        f"{child['type']}_{child['start_byte']}_{child['end_byte']}"
                    )
                    scope_manager.add_import(import_name, import_id)

        # Check for control flow nodes
        if node["type"] in PYTHON_CONTROL_FLOW_NODES:
            scope_manager.enter_control_flow(node_id)

            # Add control flow edges between blocks
            body_node = None
            for child in node.get("children", []):
                if child["type"] == "block":
                    body_node = child
                    break

            if body_node:
                # Add control flow edge from this node to its body
                edges.append(
                    {
                        "source": node_id,
                        "target": f"{body_node['type']}_{body_node['start_byte']}_{body_node['end_byte']}",
                        "type": "control_flow",
                    }
                )

        # Process all children recursively
        for child in node.get("children", []):
            find_enhanced_definitions(child, current_scope)

        # Exit any control flow blocks we entered
        if node["type"] in PYTHON_CONTROL_FLOW_NODES:
            scope_manager.exit_control_flow()

        # Restore previous scope if we created a new one
        if node["type"] in PYTHON_SCOPE_NODES:
            current_scope = old_scope

    # Second pass: find all references and connect the edges
    def find_enhanced_references(node, scope=None):
        nonlocal current_scope
        old_scope = current_scope
        node_id = f"{node['type']}_{node['start_byte']}_{node['end_byte']}"

        # Update scope if needed
        if node["type"] in PYTHON_SCOPE_NODES:
            current_scope = node_id

        # Look for references to functions, variables, etc.
        if node["type"] == "call":
            # Find the function name (first child is usually the function being called)
            func_node = None
            for child in node.get("children", []):
                if child["type"] == "identifier":
                    func_node = child
                    break

            if func_node:
                func_name = func_node["text"]
                caller_id = f"{func_node['type']}_{func_node['start_byte']}_{func_node['end_byte']}"

                # Look for the function definition
                func_id = scope_manager.find_function(func_name)
                if func_id:
                    edges.append(
                        {"source": caller_id, "target": func_id, "type": "calls"}
                    )

                # Check if it's an imported function
                import_id = scope_manager.find_import(func_name)
                if import_id:
                    edges.append(
                        {
                            "source": caller_id,
                            "target": import_id,
                            "type": "calls_import",
                        }
                    )

        elif node["type"] == "identifier":
            # Check if this is a variable reference (not a definition)
            parent_type = None
            if node.get("parent"):
                parent_type = node["parent"]["type"]

            # Skip if this is a definition (handled in the first pass)
            if parent_type not in [
                "function_definition",
                "class_definition",
                "parameter",
            ]:
                var_name = node["text"]
                var_id = node_id

                # Look for the variable definition
                ref_id = scope_manager.find_variable(var_name, current_scope)
                if ref_id and ref_id != var_id:  # Don't link to self
                    edges.append(
                        {"source": var_id, "target": ref_id, "type": "references"}
                    )

        # Process all children recursively
        for child in node.get("children", []):
            find_enhanced_references(child, current_scope)

        # Restore previous scope if we created a new one
        if node["type"] in PYTHON_SCOPE_NODES:
            current_scope = old_scope

    # Run both passes
    find_enhanced_definitions(ast)
    find_enhanced_references(ast)


def add_enhanced_js_ts_semantic_edges(ast: Dict, edges: List[Dict]):
    """
    Add enhanced JavaScript/TypeScript-specific semantic edges to the ASG.

    Similar to the Python version, but adapted for JS/TS syntax.

    Args:
        ast: The JS/TS AST
        edges: List to store the detected edges
    """
    # This would contain JS/TS-specific scope and edge analysis
    # Similar implementation as the Python version but adapted for JS/TS
    # Since this is a more advanced implementation, we're keeping it as a placeholder
    pass


def generate_ast_diff(
    ast_old: Dict, ast_new: Dict, source_old: str, source_new: str
) -> Dict:
    """
    Generate a diff between two ASTs, showing only the changed nodes.

    This is useful for incremental updates, where only the changed parts
    of the AST need to be processed, saving time and memory for large files.

    Args:
        ast_old: Old AST data
        ast_new: New AST data
        source_old: Old source code
        source_new: New source code

    Returns:
        Dictionary with the changed nodes and metadata
    """
    # If we don't have tree objects, parse both files (can't use incremental parsing)
    if "tree_object" not in ast_old or "tree_object" not in ast_new:
        return {"error": "Both ASTs must have tree_object property for diffing"}

    old_tree = ast_old["tree_object"]
    new_tree = ast_new["tree_object"]

    # Get changed ranges from Tree-sitter
    # old_source_bytes and new_source_bytes are not needed here
    # as the tree objects are already created from byte strings.

    # Get the changed ranges
    changed_ranges = []
    for edit in new_tree.get_changed_ranges(old_tree):
        changed_ranges.append(
            {
                "start_byte": edit.start_byte,
                "end_byte": edit.end_byte,
                "start_point": {
                    "row": edit.start_point[0],
                    "column": edit.start_point[1],
                },
                "end_point": {"row": edit.end_point[0], "column": edit.end_point[1]},
            }
        )

    # Find nodes in the new AST that are in the changed ranges
    changed_nodes = []

    def find_nodes_in_range(node, ranges):
        # Check if this node is in any of the changed ranges
        node_start = node["start_byte"]
        node_end = node["end_byte"]

        for r in ranges:
            # If there's any overlap between the node and the range
            if not (node_end <= r["start_byte"] or node_start >= r["end_byte"]):
                changed_nodes.append(node)
                return True

        # If node is not changed, check its children
        for child in node.get("children", []):
            if find_nodes_in_range(child, ranges):
                return True

        return False

    # Start from the root
    find_nodes_in_range(ast_new["ast"], changed_ranges)

    return {
        "language": ast_new["language"],
        "changed_ranges": changed_ranges,
        "changed_nodes": changed_nodes,
        "old_ast": ast_old["ast"],
        "new_ast": ast_new["ast"],
    }


def get_node_by_position(ast: Dict, line: int, column: int) -> Optional[Dict]:
    """
    Find the most specific node at a given line and column position.

    This is useful for pinpointing a specific location in the code,
    for example to find what function or variable is at the cursor position.

    Args:
        ast: The AST data
        line: Line number (0-based)
        column: Column number (0-based)

    Returns:
        The node at the given position, or None if not found
    """

    def find_node(node):
        # Check if the position is within this node's range
        if node["start_point"]["row"] <= line <= node["end_point"]["row"]:
            # If on start or end line, check column as well
            if (
                node["start_point"]["row"] == line
                and column < node["start_point"]["column"]
            ):
                return None
            if (
                node["end_point"]["row"] == line
                and column > node["end_point"]["column"]
            ):
                return None

            # Position is within this node, check children for more specific match
            best_match = node

            for child in node.get("children", []):
                child_match = find_node(child)
                if child_match is not None:
                    # Child contains the position, its more specific than current node
                    best_match = child_match

            return best_match

        return None

    return find_node(ast["ast"])


def register_enhanced_tools(mcp_server):
    """Register all enhanced tools with the MCP server."""

    @mcp_server.tool(name="parse_to_ast_incremental")
    def parse_to_ast_incremental(
        code: str,
        old_code: Optional[str] = None,
        language: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict:
        """
        Parse code into an AST with incremental parsing support.

        This tool can use the previous version of the code to only parse
        the parts that changed, which is much faster for large files.

        Args:
            code: New source code to parse
            old_code: Previous version of the code (optional)
            language: Programming language (e.g., 'python', 'javascript')
                     If not provided, the tool will attempt to detect it
            filename: Optional filename to help with language detection

        Returns:
            A dictionary containing the AST and language information,
            along with diff information if old_code was provided
        """
        # If old_code is provided, try to use it for incremental parsing
        previous_tree = None
        if old_code:
            # First parse the old code to get a tree
            old_result = parse_code_to_ast_incremental(
                old_code, language=language, filename=filename
            )
            if "error" not in old_result and "tree_object" in old_result:
                previous_tree = old_result["tree_object"]

        # Parse the new code, potentially using the previous tree
        return parse_code_to_ast_incremental(
            code, language, filename, previous_tree, old_code
        )

    @mcp_server.tool()
    def generate_enhanced_asg(
        code: str, language: Optional[str] = None, filename: Optional[str] = None
    ) -> Dict:
        """
        Generate an enhanced Abstract Semantic Graph (ASG) from code.

        This tool creates a more complete semantic graph with better
        scope handling, control flow edges, and data flow edges.

        Args:
            code: The source code to analyze
            language: The programming language (e.g., 'python', 'javascript')
                     If not provided, the tool will attempt to detect it
            filename: Optional filename to help with language detection

        Returns:
            A dictionary containing the enhanced ASG with nodes, edges, and metadata
        """
        ast_data = parse_to_ast_incremental(
            code, old_code=None, language=language, filename=filename
        )
        return create_enhanced_asg_from_ast(ast_data)

    @mcp_server.tool()
    def diff_ast(
        old_code: str,
        new_code: str,
        language: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict:
        """
        Compare two versions of code and return only the changed AST nodes.

        This is useful for incremental updates, where only the changed parts
        need to be analyzed, saving time and memory for large files.

        Args:
            old_code: Previous version of the code
            new_code: New version of the code
            language: Programming language (e.g., 'python', 'javascript')
            filename: Optional filename to help with language detection

        Returns:
            A dictionary with the changed nodes and metadata
        """
        ast_old = parse_to_ast_incremental(
            old_code, language=language, filename=filename
        )
        ast_new = parse_to_ast_incremental(
            new_code, language=language, filename=filename
        )

        if "error" in ast_old:
            return ast_old
        if "error" in ast_new:
            return ast_new

        return generate_ast_diff(ast_old, ast_new, old_code, new_code)

    @mcp_server.tool()
    def find_node_at_position(
        code: str,
        line: int,
        column: int,
        language: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict:
        """
        Find the AST node at a specific position in the code.

        This is useful for pinpointing a specific location in the code,
        for example to find what function or variable is at the cursor position.

        Args:
            code: The source code
            line: Line number (0-based)
            column: Column number (0-based)
            language: Programming language (e.g., 'python', 'javascript')
            filename: Optional filename to help with language detection

            Returns:
                The node at the given position, or an error if not found
        """
        ast_data = parse_to_ast_incremental(code, language=language, filename=filename)

        if "error" in ast_data:
            return ast_data

        node = get_node_by_position(ast_data, line, column)

        if node:
            return {"node": node, "language": ast_data["language"]}
        else:
            return {"error": f"No node found at position {line}:{column}"}

    return None

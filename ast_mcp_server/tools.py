"""
AST/ASG analysis tools for the MCP server.

This module defines the tools that provide code structure and semantic analysis
capabilities through the Model Context Protocol.
"""

import importlib
import os
from typing import Any, Dict, List, Optional

from tree_sitter import Language, Node, Parser

# Language modules to import
LANGUAGE_MODULES = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
}

# Path to the parsers availability marker
PARSERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parsers")
PARSERS_AVAILABLE_FILE = os.path.join(PARSERS_DIR, "parsers_available.txt")

# Language identifiers mapping
LANGUAGE_MAP = {
    "py": "python",
    "python": "python",
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "tsx": "tsx",
    "go": "go",
    "golang": "go",
    "rs": "rust",
    "rust": "rust",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "java": "java",
}

# Initialize parser and languages
languages = {}


def init_parsers() -> bool:
    """Initialize the tree-sitter parsers.

    Returns:
        True if at least one parser was initialized, False otherwise.
    """
    global languages

    # Check if parsers are marked as available
    if not os.path.exists(PARSERS_AVAILABLE_FILE):
        return False

    try:
        # Check which language modules are available
        available_languages = []
        for lang_name, module_name in LANGUAGE_MODULES.items():
            try:
                module = importlib.import_module(module_name)
                languages[lang_name] = Language(module.language())
                available_languages.append(lang_name)
            except ImportError:
                print(
                    f"Module {module_name} not found. Some language support may be limited."
                )
            except Exception as e:
                print(f"Error initializing {lang_name} language: {e}")

        if available_languages:
            print(
                f"Successfully initialized parsers for: {', '.join(available_languages)}"
            )
            return True
        else:
            return False

    except Exception as e:
        print(f"Error initializing parsers: {e}")
        return False


def node_to_dict(
    node: Node, source_bytes: bytes, include_children: bool = True
) -> Dict:
    """Convert a tree-sitter Node to a dictionary representation."""
    result = {
        "type": node.type,
        "start_byte": node.start_byte,
        "end_byte": node.end_byte,
        "start_point": {"row": node.start_point[0], "column": node.start_point[1]},
        "end_point": {"row": node.end_point[0], "column": node.end_point[1]},
        "text": source_bytes[node.start_byte : node.end_byte].decode("utf-8"),
    }

    if include_children and node.child_count > 0:
        result["children"] = [
            node_to_dict(child, source_bytes, include_children)
            for child in node.children
        ]

    return result


def create_field_edges(node: Dict, parent_id: Optional[str] = None) -> List[Dict]:
    """Create field edges for the ASG (connecting nodes with their named fields)."""
    edges = []
    node_id = f"{node['type']}_{node['start_byte']}_{node['end_byte']}"

    if parent_id:
        edges.append({"source": parent_id, "target": node_id, "type": "contains"})

    if "children" in node:
        for child in node.get("children", []):
            edges.extend(create_field_edges(child, node_id))

    return edges


def detect_language(code: str, filename: Optional[str] = None) -> str:
    """Detect the programming language from code content and/or filename."""
    if filename:
        ext = filename.split(".")[-1].lower()
        if ext in LANGUAGE_MAP:
            return LANGUAGE_MAP[ext]

    # Simple heuristics for language detection
    if "def " in code and ":" in code and "import " in code:
        return "python"
    elif "func " in code and "{" in code and "}" in code and "package " in code:
        return "go"
    elif "fn " in code and "let " in code and "->" in code:
        return "rust"
    elif "const " in code and "function" in code and "=>" in code:
        return "typescript"
    elif "function" in code and "var " in code and ";" in code:
        return "javascript"
    elif "class " in code and "public " in code and "void " in code:
        return "java"
    elif "int " in code and "#include" in code:
        return "c"
    elif "std::" in code or "template<" in code:
        return "cpp"

    # Default to Python if we can't detect
    return "python"


def parse_code_to_ast(
    code: str,
    language: Optional[str] = None,
    filename: Optional[str] = None,
    include_children: bool = True,
) -> Dict:
    """
    Parse code into an Abstract Syntax Tree (AST) using tree-sitter.

    Args:
        code: Source code to parse
        language: Programming language identifier (optional)
        filename: Source file name (optional, used for language detection)
        include_children: Whether to include child nodes in the result

    Returns:
        Dictionary representation of the AST
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
        # Create a parser and set the language
        parser = Parser()
        parser.language = languages[language]

        # Parse the code
        source_bytes = bytes(code, "utf-8")
        tree = parser.parse(source_bytes)

        # Convert to dictionary
        root_node = tree.root_node
        ast = node_to_dict(root_node, source_bytes, include_children)

        return {"language": language, "ast": ast}
    except Exception as e:
        return {"error": f"Error parsing code: {e}"}


def create_asg_from_ast(ast_data: Dict) -> Dict:
    """
    Create an Abstract Semantic Graph (ASG) from an AST.

    This is a simplified version that extracts some basic semantic information.
    A production version would have more sophisticated analysis.

    Args:
        ast_data: AST data from parse_code_to_ast

    Returns:
        Dictionary representation of the ASG
    """
    if "error" in ast_data:
        return ast_data

    ast = ast_data["ast"]
    language = ast_data["language"]

    # Extract nodes and edges from the AST
    nodes = []
    edges = []

    def extract_nodes(node: Dict, parent_id: Optional[str] = None) -> str:
        node_id = f"{node['type']}_{node['start_byte']}_{node['end_byte']}"

        # Add the node
        nodes.append(
            {
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
        )

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
        add_python_semantic_edges(ast, edges)
    elif language in ["javascript", "typescript"]:
        add_js_ts_semantic_edges(ast, edges)

    return {"language": language, "nodes": nodes, "edges": edges, "root": root_id}


def add_python_semantic_edges(ast: Dict[str, Any], edges: List[Dict[str, Any]]) -> None:
    """Add Python-specific semantic edges to the ASG.

    This is a simplified implementation that extracts function calls
    and variable references.

    Args:
        ast: The parsed AST dictionary
        edges: List to append semantic edges to
    """
    # This is a simplified implementation for demo purposes
    # A real implementation would do much deeper analysis

    # Find all function definitions
    functions: Dict[str, str] = {}
    variables: Dict[Optional[str], Dict[str, str]] = {}

    def find_definitions(node: Dict[str, Any], scope: Optional[str] = None) -> None:
        """Find and record function and variable definitions."""
        if node["type"] == "function_definition":
            # Get function name
            for child in node.get("children", []):
                if child["type"] == "identifier":
                    func_name = child["text"]
                    func_id = f"{node['type']}_{node['start_byte']}_{node['end_byte']}"
                    functions[func_name] = func_id

                    # New scope for this function
                    new_scope = func_id

                    # Process the function body with this new scope
                    for body_child in node.get("children", []):
                        if body_child["type"] == "block":
                            find_definitions(body_child, new_scope)

        elif node["type"] == "assignment":
            # Track variable assignments
            for child in node.get("children", []):
                if child["type"] == "identifier":
                    var_name = child["text"]
                    var_id = (
                        f"{child['type']}_{child['start_byte']}_{child['end_byte']}"
                    )

                    # Store in current scope
                    if scope not in variables:
                        variables[scope] = {}
                    variables[scope][var_name] = var_id

        # Process all children
        for child in node.get("children", []):
            find_definitions(child, scope)

    # Start analysis from the root
    find_definitions(ast)

    # Now scan for references
    def find_references(node: Dict[str, Any], scope: Optional[str] = None) -> None:
        """Find and record references to functions and variables."""
        if node["type"] == "call":
            # Check for function calls
            for child in node.get("children", []):
                if child["type"] == "identifier":
                    func_name = child["text"]
                    if func_name in functions:
                        caller_id = (
                            f"{child['type']}_{child['start_byte']}_{child['end_byte']}"
                        )
                        edges.append(
                            {
                                "source": caller_id,
                                "target": functions[func_name],
                                "type": "calls",
                            }
                        )

        elif node["type"] == "identifier":
            # Check for variable references
            var_name = node["text"]
            var_id = f"{node['type']}_{node['start_byte']}_{node['end_byte']}"

            # Check in current scope first, then parent scopes
            current_scope = scope
            while current_scope is not None:
                if current_scope in variables and var_name in variables[current_scope]:
                    target_id = variables[current_scope][var_name]
                    if var_id != target_id:  # Don't link to self
                        edges.append(
                            {
                                "source": var_id,
                                "target": target_id,
                                "type": "references",
                            }
                        )
                    break

                # Move up to parent scope
                # This is simplistic - real implementation would track scope hierarchy
                current_scope = None

        # Process all children
        for child in node.get("children", []):
            find_references(child, scope)

    # Start reference analysis from the root
    find_references(ast)


def add_js_ts_semantic_edges(ast: Dict[str, Any], edges: List[Dict[str, Any]]) -> None:
    """Add JavaScript/TypeScript-specific semantic edges to the ASG.

    Similar to Python version but adapted for JS/TS syntax.
    Currently a placeholder for future implementation.

    Args:
        ast: The parsed AST dictionary
        edges: List to append semantic edges to
    """
    # Similar to Python version but adapted for JS/TS syntax
    # This is also a simplified implementation for demo purposes

    # Functionality similar to the Python version would go here
    # Since this is a demo, we're keeping it simple
    pass


def analyze_code_structure(
    code: str, language: Optional[str] = None, filename: Optional[str] = None
) -> Dict:
    """
    Analyze code structure and provide insights.

    Args:
        code: Source code to analyze
        language: Programming language identifier (optional)
        filename: Source file name (optional)

    Returns:
        Dictionary with code structure analysis
    """
    # Parse code to AST
    ast_data = parse_code_to_ast(code, language, filename)
    if "error" in ast_data:
        return ast_data

    # Perform analysis based on the AST
    language = ast_data["language"]
    ast = ast_data["ast"]

    # Collect structure information
    structure: Dict[str, Any] = {
        "language": language,
        "code_length": len(code),
        "functions": [],
        "classes": [],
        "imports": [],
        "complexity_metrics": {"max_nesting_level": 0, "total_nodes": 0},
    }

    # Calculate metrics based on language
    if language == "python":
        analyze_python_structure(ast, structure)
    elif language in ["javascript", "typescript"]:
        analyze_js_ts_structure(ast, structure)
    elif language == "go":
        analyze_go_structure(ast, structure)
    # Add more language analyzers as needed

    return structure


def analyze_python_structure(ast: Dict[str, Any], structure: Dict[str, Any]) -> None:
    """Analyze Python code structure.

    Extracts functions, classes, imports and complexity metrics.

    Args:
        ast: The parsed AST dictionary
        structure: Output dictionary to populate with analysis results
    """
    # Track functions
    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    imports: List[Dict[str, Any]] = []

    def count_nodes(node: Dict[str, Any]) -> int:
        """Count total nodes in the AST."""
        count = 1  # Count this node
        for child in node.get("children", []):
            count += count_nodes(child)
        return count

    def find_max_nesting(node: Dict[str, Any], current_depth: int = 0) -> int:
        """Find the maximum nesting level in the AST."""
        max_depth = current_depth

        # Increase depth for nesting structures
        if node["type"] in [
            "if_statement",
            "for_statement",
            "while_statement",
            "try_statement",
            "with_statement",
        ]:
            current_depth += 1
            max_depth = current_depth

        # Check children
        for child in node.get("children", []):
            child_max = find_max_nesting(child, current_depth)
            max_depth = max(max_depth, child_max)

        return max_depth

    def extract_structures(node: Dict[str, Any]) -> None:
        """Extract function, class, and import structures from the AST."""
        if node["type"] == "function_definition":
            # Extract function name
            name = ""
            for child in node.get("children", []):
                if child["type"] == "identifier":
                    name = child["text"]
                    break

            # Get function parameters
            params = []
            for child in node.get("children", []):
                if child["type"] == "parameters":
                    for param_child in child.get("children", []):
                        if param_child["type"] == "identifier":
                            params.append(param_child["text"])

            functions.append(
                {
                    "name": name,
                    "location": {
                        "start_line": node["start_point"]["row"] + 1,
                        "end_line": node["end_point"]["row"] + 1,
                    },
                    "parameters": params,
                }
            )

        elif node["type"] == "class_definition":
            # Extract class name
            name = ""
            for child in node.get("children", []):
                if child["type"] == "identifier":
                    name = child["text"]
                    break

            classes.append(
                {
                    "name": name,
                    "location": {
                        "start_line": node["start_point"]["row"] + 1,
                        "end_line": node["end_point"]["row"] + 1,
                    },
                }
            )

        elif (
            node["type"] == "import_statement"
            or node["type"] == "import_from_statement"
        ):
            # Get imported module names
            module_names = []
            for child in node.get("children", []):
                if child["type"] == "dotted_name":
                    module_names.append(child["text"])

            imports.append(
                {
                    "module": ".".join(module_names),
                    "line": node["start_point"]["row"] + 1,
                }
            )

        # Process children
        for child in node.get("children", []):
            extract_structures(child)

    # Process the AST
    extract_structures(ast)

    # Count total nodes
    total_nodes = count_nodes(ast)

    # Find maximum nesting level
    max_nesting = find_max_nesting(ast)

    # Update structure
    structure["functions"] = functions
    structure["classes"] = classes
    structure["imports"] = imports
    structure["complexity_metrics"]["total_nodes"] = total_nodes
    structure["complexity_metrics"]["max_nesting_level"] = max_nesting


def analyze_js_ts_structure(ast: Dict[str, Any], structure: Dict[str, Any]) -> None:
    """Analyze JavaScript/TypeScript code structure.

    Similar implementation as the Python version but adapted for JS/TS.
    Currently a placeholder for future implementation.

    Args:
        ast: The parsed AST dictionary
        structure: Output dictionary to populate with analysis results
    """
    # Similar implementation as the Python version but adapted for JS/TS
    # Since this is a demo, we're keeping it simplified
    pass


def analyze_go_structure(ast: Dict[str, Any], structure: Dict[str, Any]) -> None:
    """Analyze Go code structure.

    Similar implementation as the Python version but adapted for Go.
    Currently a placeholder for future implementation.

    Args:
        ast: The parsed AST dictionary
        structure: Output dictionary to populate with analysis results
    """
    # Similar implementation as the Python version but adapted for Go
    # Since this is a demo, we're keeping it simplified
    pass


def register_tools(mcp_server: Any) -> None:
    """Register all tools with the MCP server.

    Args:
        mcp_server: The FastMCP server instance to register tools with
    """

    @mcp_server.tool()
    def parse_to_ast(
        code: str, language: Optional[str] = None, filename: Optional[str] = None
    ) -> Dict:
        """
        Parse code into an Abstract Syntax Tree (AST).

        This tool takes source code and returns its syntax tree representation.
        The AST provides structural information about the code, showing how
        different parts of the code relate to each other syntactically.

        Args:
            code: The source code to parse
            language: The programming language (e.g., 'python', 'javascript')
                     If not provided, the tool will attempt to detect it
            filename: Optional filename to help with language detection

        Returns:
            A dictionary containing the AST and language information
        """
        return parse_code_to_ast(code, language, filename)

    @mcp_server.tool()
    def generate_asg(
        code: str, language: Optional[str] = None, filename: Optional[str] = None
    ) -> Dict:
        """
        Generate an Abstract Semantic Graph (ASG) from code.

        This tool analyzes code and creates a semantic graph that captures not just
        syntax but also semantic relationships like variable references, function calls,
        type information, and data flow where possible.

        Args:
            code: The source code to analyze
            language: The programming language (e.g., 'python', 'javascript')
                     If not provided, the tool will attempt to detect it
            filename: Optional filename to help with language detection

        Returns:
            A dictionary containing the ASG nodes, edges, and metadata
        """
        ast_data = parse_code_to_ast(code, language, filename)
        return create_asg_from_ast(ast_data)

    @mcp_server.tool()
    def analyze_code(
        code: str, language: Optional[str] = None, filename: Optional[str] = None
    ) -> Dict:
        """
        Analyze code structure and provide insights.

        This tool examines code and extracts structural information like functions,
        classes, complexity metrics, and other language-specific details.

        Args:
            code: The source code to analyze
            language: The programming language (e.g., 'python', 'javascript')
                     If not provided, the tool will attempt to detect it
            filename: Optional filename to help with language detection

        Returns:
            A dictionary with analysis results including structure and metrics
        """
        return analyze_code_structure(code, language, filename)

    @mcp_server.tool()
    def supported_languages() -> List[str]:
        """
        Get the list of supported programming languages.

        Returns:
            A list of programming language identifiers that can be parsed
        """
        # Initialize parsers if not done already
        if not languages and not init_parsers():
            return []

        return list(languages.keys())

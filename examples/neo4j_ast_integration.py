#!/usr/bin/env python
"""
Neo4j Integration for AST/ASG Analysis

This script demonstrates how to integrate AST/ASG analysis with Neo4j
for advanced code structure queries and analysis.
"""

import hashlib
import os
import sys
from typing import Any, Dict, List, Optional

# Add the parent directory to the path so we can import the tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ast_mcp_server.tools import (
    analyze_code_structure,
    create_asg_from_ast,
    init_parsers,
    parse_code_to_ast,
)

# Try to import Neo4j driver
try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None  # Define GraphDatabase as None when import fails
    print("Warning: Neo4j driver not available. Install with 'pip install neo4j'")


class AstNeo4jIntegration:
    """Integration class for AST/ASG analysis with Neo4j."""

    # trunk-ignore(bandit/B107)
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        db: str = "neo4j",
    ) -> None:
        """Initialize the Neo4j connection."""
        self.uri = uri
        self.user = user
        self.password = password
        self.db = db
        self.driver = None
        if NEO4J_AVAILABLE and GraphDatabase is not None:
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                # Test connection
                with self.driver.session(database=db) as session:
                    result = session.run("RETURN 1 AS test")
                    record = result.single()
                    if record is not None:
                        test_value = record["test"]
                        if test_value == 1:
                            print(f"✅ Connected to Neo4j at {uri}")
                        else:
                            print("❌ Connection test failed")
                    else:
                        print("❌ Connection test failed - no result returned")
            except Exception as e:
                print(f"❌ Failed to connect to Neo4j: {e}")
                self.driver = None
        else:
            print("⚠️ Neo4j integration disabled (driver not available)")

    def store_ast_in_neo4j(
        self, ast_data: Dict[str, Any], file_path: str
    ) -> Optional[None]:
        """
        Store AST data in Neo4j for querying.

        Args:
            ast_data: AST result from parse_code_to_ast
            file_path: Path to the source file
        """
        if not NEO4J_AVAILABLE or not self.driver:
            return None

        if "error" in ast_data:
            print(f"⚠️ Cannot store AST with error: {ast_data['error']}")
            return None

        # Generate a unique ID for this AST
        file_name = os.path.basename(file_path)
        ast_id = hashlib.md5(
            f"{file_path}:{ast_data['language']}".encode(), usedforsecurity=False
        ).hexdigest()

        # Create statements
        with self.driver.session(database=self.db) as session:
            # Create file node
            session.run(
                """
                MERGE (f:SourceFile {path: $path, name: $name})
                SET f.language = $language
                RETURN f
                """,
                path=file_path,
                name=file_name,
                language=ast_data["language"],
            )

            # Create AST node
            session.run(
                """
                MATCH (f:SourceFile {path: $path})
                MERGE (ast:AST {id: $ast_id})
                SET ast.language = $language
                MERGE (f)-[:HAS_AST]->(ast)
                RETURN ast
                """,
                path=file_path,
                ast_id=ast_id,
                language=ast_data["language"],
            )

    def _add_ast_node_to_neo4j(
        self, session: Any, ast_id: str, parent_id: Optional[str], node: Dict[str, Any]
    ) -> None:
        """Add an AST node to Neo4j recursively."""
        # Generate a unique ID for this node
        node_id = f"{ast_id}_{node['type']}_{node['start_byte']}_{node['end_byte']}"

        # Create the node
        session.run(
            """
            MATCH (ast:AST {id: $ast_id})
            MERGE (n:ASTNode {id: $node_id})
            SET n.type = $type,
                n.text = $text,
                n.start_byte = $start_byte,
                n.end_byte = $end_byte,
                n.start_line = $start_line,
                n.start_col = $start_col,
                n.end_line = $end_line,
                n.end_col = $end_col
            MERGE (ast)-[:CONTAINS]->(n)
            """,
            ast_id=ast_id,
            node_id=node_id,
            type=node["type"],
            text=node["text"],
            start_byte=node["start_byte"],
            end_byte=node["end_byte"],
            start_line=node["start_point"]["row"],
            start_col=node["start_point"]["column"],
            end_line=node["end_point"]["row"],
            end_col=node["end_point"]["column"],
        )

        # Link to parent if exists
        if parent_id:
            session.run(
                """
                MATCH (p:ASTNode {id: $parent_id})
                MATCH (n:ASTNode {id: $node_id})
                MERGE (p)-[:HAS_CHILD]->(n)
                """,
                parent_id=parent_id,
                node_id=node_id,
            )

        # Process children
        if "children" in node:
            for child in node["children"]:
                self._add_ast_node_to_neo4j(session, ast_id, node_id, child)

    def store_asg_in_neo4j(
        self, asg_data: Dict[str, Any], file_path: str
    ) -> Optional[str]:
        """
        Store ASG data in Neo4j for querying.

        Args:
            asg_data: ASG result from create_asg_from_ast
            file_path: Path to the source file
        """
        if not NEO4J_AVAILABLE or not self.driver:
            return None

        if "error" in asg_data:
            print(f"⚠️ Cannot store ASG with error: {asg_data['error']}")
            return None

        # Generate a unique ID for this ASG
        file_name = os.path.basename(file_path)
        asg_id = hashlib.md5(
            f"{file_path}:{asg_data['language']}:asg".encode(), usedforsecurity=False
        ).hexdigest()

        with self.driver.session(database=self.db) as session:
            # Create file node if not exists
            session.run(
                """
                MERGE (f:SourceFile {path: $path, name: $name})
                SET f.language = $language
                RETURN f
                """,
                path=file_path,
                name=file_name,
                language=asg_data["language"],
            )

            # Create ASG node
            session.run(
                """
                MATCH (f:SourceFile {path: $path})
                MERGE (asg:ASG {id: $asg_id})
                SET asg.language = $language
                MERGE (f)-[:HAS_ASG]->(asg)
                """,
                path=file_path,
                asg_id=asg_id,
                language=asg_data["language"],
            )

            # Add nodes
            for node in asg_data["nodes"]:
                session.run(
                    """
                    MATCH (asg:ASG {id: $asg_id})
                    MERGE (n:ASGNode {id: $node_id})
                    SET n.type = $type,
                        n.text = $text,
                        n.start_byte = $start_byte,
                        n.end_byte = $end_byte,
                        n.start_line = $start_line,
                        n.start_col = $start_col,
                        n.end_line = $end_line,
                        n.end_col = $end_col
                    MERGE (asg)-[:CONTAINS]->(n)
                    """,
                    asg_id=asg_id,
                    node_id=node["id"],
                    type=node["type"],
                    text=node["text"],
                    start_byte=node.get("start_byte", 0),
                    end_byte=node.get("end_byte", 0),
                    start_line=node.get("start_line", 0),
                    start_col=node.get("start_col", 0),
                    end_line=node.get("end_line", 0),
                    end_col=node.get("end_col", 0),
                )

            # Add edges
            for edge in asg_data["edges"]:
                session.run(
                    """
                    MATCH (s:ASGNode {id: $source_id})
                    MATCH (t:ASGNode {id: $target_id})
                    MERGE (s)-[r:EDGE {type: $edge_type}]->(t)
                    """,
                    source_id=edge["source"],
                    target_id=edge["target"],
                    edge_type=edge["type"],
                )

            print(f"✅ Stored ASG in Neo4j with ID: {asg_id}")
            return asg_id

    def store_analysis_in_neo4j(
        self, analysis_data: Dict[str, Any], file_path: str
    ) -> Optional[str]:
        """
        Store code analysis results in Neo4j.

        Args:
            analysis_data: Analysis result from analyze_code_structure
            file_path: Path to the source file
        """
        if not NEO4J_AVAILABLE or not self.driver:
            return None

        if "error" in analysis_data:
            print(f"⚠️ Cannot store analysis with error: {analysis_data['error']}")
            return None

        # Generate a unique ID for the analysis
        file_name = os.path.basename(file_path)
        analysis_id = hashlib.md5(
            f"{file_path}:{analysis_data['language']}:analysis".encode(),
            usedforsecurity=False,
        ).hexdigest()

        with self.driver.session(database=self.db) as session:
            # Create file node if not exists
            session.run(
                """
                MERGE (f:SourceFile {path: $path, name: $name})
                SET f.language = $language
                RETURN f
                """,
                path=file_path,
                name=file_name,
                language=analysis_data["language"],
            )

            # Create CodeAnalysis node
            session.run(
                """
                MATCH (f:SourceFile {path: $path})
                MERGE (a:CodeAnalysis {id: $analysis_id})
                SET a.language = $language,
                    a.code_length = $code_length,
                    a.max_nesting_level = $max_nesting,
                    a.total_nodes = $total_nodes
                MERGE (f)-[:HAS_ANALYSIS]->(a)
                """,
                path=file_path,
                analysis_id=analysis_id,
                language=analysis_data["language"],
                code_length=analysis_data["code_length"],
                max_nesting=analysis_data["complexity_metrics"]["max_nesting_level"],
                total_nodes=analysis_data["complexity_metrics"]["total_nodes"],
            )

            # Add functions
            for func in analysis_data["functions"]:
                func_id = hashlib.md5(
                    f"{analysis_id}:func:{func['name']}:{func['location']['start_line']}".encode(),
                    usedforsecurity=False,
                ).hexdigest()
                session.run(
                    """
                    MATCH (a:CodeAnalysis {id: $analysis_id})
                    MERGE (f:Function {id: $func_id})
                    SET f.name = $name,
                        f.start_line = $start_line,
                        f.end_line = $end_line,
                        f.parameters = $parameters
                    MERGE (a)-[:HAS_FUNCTION]->(f)
                    """,
                    analysis_id=analysis_id,
                    func_id=func_id,
                    name=func["name"],
                    start_line=func["location"]["start_line"],
                    end_line=func["location"]["end_line"],
                    parameters=func["parameters"],
                )

            # Add classes
            for cls in analysis_data["classes"]:
                cls_id = hashlib.md5(
                    f"{analysis_id}:class:{cls['name']}:{cls['location']['start_line']}".encode(),
                    usedforsecurity=False,
                ).hexdigest()
                session.run(
                    """
                    MATCH (a:CodeAnalysis {id: $analysis_id})
                    MERGE (c:Class {id: $cls_id})
                    SET c.name = $name,
                        c.start_line = $start_line,
                        c.end_line = $end_line
                    MERGE (a)-[:HAS_CLASS]->(c)
                    """,
                    analysis_id=analysis_id,
                    cls_id=cls_id,
                    name=cls["name"],
                    start_line=cls["location"]["start_line"],
                    end_line=cls["location"]["end_line"],
                )

            # Add imports
            for imp in analysis_data["imports"]:
                imp_id = hashlib.md5(
                    f"{analysis_id}:import:{imp['module']}:{imp['line']}".encode(),
                    usedforsecurity=False,
                ).hexdigest()
                session.run(
                    """
                    MATCH (a:CodeAnalysis {id: $analysis_id})
                    MERGE (i:Import {id: $imp_id})
                    SET i.module = $module,
                        i.line = $line
                    MERGE (a)-[:HAS_IMPORT]->(i)
                    """,
                    analysis_id=analysis_id,
                    imp_id=imp_id,
                    module=imp["module"],
                    line=imp["line"],
                )

            print(f"✅ Stored code analysis in Neo4j with ID: {analysis_id}")
            return analysis_id

    def find_complex_functions(
        self, nesting_threshold: int = 3
    ) -> List[Dict[str, Any]]:
        """Find functions with high nesting levels."""
        if not NEO4J_AVAILABLE or not self.driver:
            return []

        with self.driver.session(database=self.db) as session:
            result = session.run(
                """
                MATCH (f:SourceFile)-[:HAS_ANALYSIS]->(a:CodeAnalysis)-[:HAS_FUNCTION]->(func:Function)
                WHERE a.max_nesting_level >= $threshold
                RETURN f.path AS file_path, func.name AS function_name,
                       func.start_line AS start_line, func.end_line AS end_line,
                       a.max_nesting_level AS nesting_level
                ORDER BY a.max_nesting_level DESC
                """,
                threshold=nesting_threshold,
            )
            return [dict(record) for record in result]

    def find_function_calls(self) -> List[Dict[str, Any]]:
        """Find function call relationships."""
        if not NEO4J_AVAILABLE or not self.driver:
            return []

        with self.driver.session(database=self.db) as session:
            result = session.run(
                """
                MATCH (f:SourceFile)-[:HAS_ASG]->(asg:ASG)-[:CONTAINS]->(caller:ASGNode)
                MATCH (caller)-[r:EDGE {type: 'calls'}]->(callee:ASGNode)
                RETURN f.path AS file_path, caller.text AS caller, callee.text AS callee,
                       caller.start_line AS caller_line, callee.start_line AS callee_line
                """
            )
            return [dict(record) for record in result]

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()


def main() -> None:
    """Main function to demonstrate Neo4j integration."""
    print("AST/ASG Neo4j Integration Example")
    print("-" * 50)

    # Check if parsers are initialized
    print("Initializing parsers...")
    parsers_available = init_parsers()
    if not parsers_available:
        print("⚠️ Tree-sitter parsers not available.")
        print("This example will still demonstrate the Neo4j integration structure.")

    # Example code file
    example_file = os.path.join(os.path.dirname(__file__), "example_code.py")

    # Create example code file if it doesn't exist
    if not os.path.exists(example_file):
        example_code = """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Calculate factorial of 5
result = factorial(5)
print(f"Factorial of 5 is {result}")
"""
        with open(example_file, "w") as f:
            f.write(example_code)
        print(f"Created example code file: {example_file}")

    # Read example code
    with open(example_file, "r") as f:
        code = f.read()

    # Initialize Neo4j integration
    neo4j_integration = AstNeo4jIntegration()

    # Parse code to AST and store in Neo4j
    ast_result = parse_code_to_ast(code, filename=example_file)
    if "error" not in ast_result:
        print("\n1. Storing AST in Neo4j")
        neo4j_integration.store_ast_in_neo4j(ast_result, example_file)
    else:
        print(f"\n⚠️ Cannot parse code to AST: {ast_result['error']}")

    # Create ASG from AST and store in Neo4j
    if "error" not in ast_result:
        print("\n2. Creating and storing ASG in Neo4j")
        asg_result = create_asg_from_ast(ast_result)
        neo4j_integration.store_asg_in_neo4j(asg_result, example_file)

    # Analyze code structure and store in Neo4j
    print("\n3. Analyzing code and storing results in Neo4j")
    analysis_result = analyze_code_structure(code, filename=example_file)
    if "error" not in analysis_result:
        neo4j_integration.store_analysis_in_neo4j(analysis_result, example_file)

    # Demonstrate Neo4j queries for code analysis
    print("\n4. Example Neo4j queries for code analysis")

    # Find complex functions
    print("\nComplex functions (nesting level >= 2):")
    complex_functions = neo4j_integration.find_complex_functions(nesting_threshold=2)
    if complex_functions:
        for func in complex_functions:
            print(
                f"  {func['function_name']} in {func['file_path']} (lines {func['start_line']}-{func['end_line']}), nesting: {func['nesting_level']}"
            )
    else:
        print("  No complex functions found (or Neo4j integration not available)")

    # Find function calls
    print("\nFunction call relationships:")
    function_calls = neo4j_integration.find_function_calls()
    if function_calls:
        for call in function_calls:
            print(
                f"  {call['caller']} (line {call['caller_line']}) calls {call['callee']} (line {call['callee_line']}) in {call['file_path']}"
            )
    else:
        print("  No function calls found (or Neo4j integration not available)")

    # Example Cypher queries
    print("\nExample Cypher queries for advanced analysis:")
    print(
        """
    // Find recursive functions
    MATCH (f:SourceFile)-[:HAS_ASG]->(asg:ASG)-[:CONTAINS]->(func:ASGNode)
    MATCH p = (func)-[:EDGE*]->(func)
    WHERE func.type = 'function_definition'
    RETURN DISTINCT f.path AS file, func.text AS function

    // Find unused imports
    MATCH (f:SourceFile)-[:HAS_AST]->(ast:AST)-[:CONTAINS]->(imp:ASTNode)
    WHERE imp.type = 'import_statement'
    OPTIONAL MATCH (f)-[:HAS_ASG]->(asg:ASG)-[:CONTAINS]->(ref:ASGNode)-[:EDGE {type: 'references'}]->()
    WHERE ref.text CONTAINS imp.text
    WITH f, imp, COUNT(ref) AS usage_count
    WHERE usage_count = 0
    RETURN f.path AS file, imp.text AS unused_import, imp.start_line AS line

    // Find code complexity metrics
    MATCH (f:SourceFile)-[:HAS_ANALYSIS]->(a:CodeAnalysis)
    RETURN f.path AS file, a.max_nesting_level AS max_nesting,
           SIZE((a)-[:HAS_FUNCTION]->()) AS function_count,
           SIZE((a)-[:HAS_CLASS]->()) AS class_count
    ORDER BY a.max_nesting_level DESC
    """
    )

    # Close Neo4j connection
    neo4j_integration.close()
    print("\nDemo completed!")


if __name__ == "__main__":
    main()

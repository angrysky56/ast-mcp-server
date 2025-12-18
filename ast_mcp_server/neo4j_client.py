"""
Neo4j client for storing and querying AST/ASG data.

This module provides the integration with a Neo4j database to persist
code analysis results, enabling complex queries and visualizations.

Uses batch insertion (UNWIND) for performance.
"""

import hashlib
import os
import sys
from typing import Any, Dict, List, Optional

# Try to import Neo4j driver
try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None


class Neo4jClient:
    """Client for interacting with Neo4j database."""

    def __init__(self) -> None:
        """Initialize the Neo4j connection using environment variables."""
        self.uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.environ.get("NEO4J_USER", "neo4j")
        self.password = os.environ.get("NEO4J_PASSWORD", "password")
        self.db = os.environ.get("NEO4J_DB", "neo4j")
        self.driver = None

        if NEO4J_AVAILABLE and GraphDatabase is not None:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri, auth=(self.user, self.password)
                )
                print(f"Initialized Neo4j driver for {self.uri}", file=sys.stderr)
            except Exception as e:
                print(f"Failed to initialize Neo4j driver: {e}", file=sys.stderr)
                self.driver = None
        else:
            print("Neo4j driver not available. Install with 'uv add neo4j'", file=sys.stderr)

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()

    def is_connected(self) -> bool:
        """Check if connected to Neo4j."""
        if not self.driver:
            return False
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def store_ast(self, ast_data: Dict[str, Any], file_path: str) -> Optional[str]:
        """
        Store AST data in Neo4j using batch insertion.

        Args:
            ast_data: AST result from parse_code_to_ast
            file_path: Path to the source file

        Returns:
            The AST ID if successful, None otherwise
        """
        if not self.driver or "error" in ast_data:
            return None

        file_name = os.path.basename(file_path)
        ast_id = hashlib.md5(
            f"{file_path}:{ast_data['language']}".encode(), usedforsecurity=False
        ).hexdigest()

        # Flatten the AST tree into lists
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, str]] = []

        def flatten_tree(node: Dict[str, Any], parent_id: Optional[str] = None) -> None:
            node_id = f"{ast_id}_{node['type']}_{node.get('start_byte', 0)}_{node.get('end_byte', 0)}"
            nodes.append(
                {
                    "id": node_id,
                    "type": node.get("type", "unknown"),
                    "text": node.get("text", "")[:500],
                    "start_byte": node.get("start_byte", 0),
                    "end_byte": node.get("end_byte", 0),
                    "start_line": node.get("start_point", {}).get("row", 0),
                    "start_col": node.get("start_point", {}).get("column", 0),
                    "end_line": node.get("end_point", {}).get("row", 0),
                    "end_col": node.get("end_point", {}).get("column", 0),
                }
            )
            if parent_id:
                edges.append({"source": parent_id, "target": node_id})
            for child in node.get("children", []):
                flatten_tree(child, node_id)

        if "ast" in ast_data:
            flatten_tree(ast_data["ast"])

        with self.driver.session(database=self.db) as session:
            # Create file and AST nodes
            session.run(
                """
                MERGE (f:SourceFile {path: $path})
                SET f.name = $name, f.language = $language
                MERGE (ast:AST {id: $ast_id})
                SET ast.language = $language
                MERGE (f)-[:HAS_AST]->(ast)
                """,
                path=file_path,
                name=file_name,
                ast_id=ast_id,
                language=ast_data.get("language", "unknown"),
            )

            # Batch insert nodes
            session.run(
                """
                MATCH (ast:AST {id: $ast_id})
                UNWIND $nodes as n
                MERGE (node:ASTNode {id: n.id})
                SET node.type = n.type,
                    node.text = n.text,
                    node.start_byte = n.start_byte,
                    node.end_byte = n.end_byte,
                    node.start_line = n.start_line,
                    node.start_col = n.start_col,
                    node.end_line = n.end_line,
                    node.end_col = n.end_col
                MERGE (ast)-[:CONTAINS]->(node)
                """,
                ast_id=ast_id,
                nodes=nodes,
            )

            # Batch insert edges (parent-child)
            session.run(
                """
                UNWIND $edges as e
                MATCH (p:ASTNode {id: e.source})
                MATCH (c:ASTNode {id: e.target})
                MERGE (p)-[:HAS_CHILD]->(c)
                """,
                edges=edges,
            )

        return ast_id

    def store_asg(self, asg_data: Dict[str, Any], file_path: str) -> Optional[str]:
        """
        Store ASG data in Neo4j using batch insertion.

        Args:
            asg_data: ASG result from create_asg_from_ast
            file_path: Path to the source file

        Returns:
            The ASG ID if successful, None otherwise
        """
        if not self.driver or "error" in asg_data:
            return None

        file_name = os.path.basename(file_path)
        asg_id = hashlib.md5(
            f"{file_path}:{asg_data['language']}:asg".encode(), usedforsecurity=False
        ).hexdigest()

        # Prepare nodes and edges for batch insert
        nodes = [
            {
                "id": n["id"],
                "type": n.get("type", "unknown"),
                "text": n.get("text", "")[:500],
                "start_byte": n.get("start_byte", 0),
                "end_byte": n.get("end_byte", 0),
            }
            for n in asg_data.get("nodes", [])
        ]

        edges = [
            {
                "source": e["source"],
                "target": e["target"],
                "type": e.get("type", "unknown"),
            }
            for e in asg_data.get("edges", [])
        ]

        with self.driver.session(database=self.db) as session:
            # Create file and ASG nodes
            session.run(
                """
                MERGE (f:SourceFile {path: $path})
                SET f.name = $name, f.language = $language
                MERGE (asg:ASG {id: $asg_id})
                SET asg.language = $language
                MERGE (f)-[:HAS_ASG]->(asg)
                """,
                path=file_path,
                name=file_name,
                asg_id=asg_id,
                language=asg_data.get("language", "unknown"),
            )

            # Batch insert nodes
            session.run(
                """
                MATCH (asg:ASG {id: $asg_id})
                UNWIND $nodes as n
                MERGE (node:ASGNode {id: n.id})
                SET node.type = n.type,
                    node.text = n.text,
                    node.start_byte = n.start_byte,
                    node.end_byte = n.end_byte
                MERGE (asg)-[:CONTAINS]->(node)
                """,
                asg_id=asg_id,
                nodes=nodes,
            )

            # Batch insert edges
            session.run(
                """
                UNWIND $edges as e
                MATCH (s:ASGNode {id: e.source})
                MATCH (t:ASGNode {id: e.target})
                MERGE (s)-[r:EDGE]->(t)
                SET r.type = e.type
                """,
                edges=edges,
            )

        return asg_id

    def store_analysis(
        self, analysis_data: Dict[str, Any], file_path: str
    ) -> Optional[str]:
        """
        Store code analysis results in Neo4j using batch insertion.

        Args:
            analysis_data: Analysis result from analyze_code_structure
            file_path: Path to the source file

        Returns:
            The analysis ID if successful, None otherwise
        """
        if not self.driver or "error" in analysis_data:
            return None

        file_name = os.path.basename(file_path)
        analysis_id = hashlib.md5(
            f"{file_path}:{analysis_data['language']}:analysis".encode(),
            usedforsecurity=False,
        ).hexdigest()

        # Prepare functions for batch insert
        functions = []
        for func in analysis_data.get("functions", []):
            func_id = hashlib.md5(
                f"{analysis_id}:func:{func['name']}:{func.get('location', {}).get('start_line', 0)}".encode(),
                usedforsecurity=False,
            ).hexdigest()
            functions.append(
                {
                    "id": func_id,
                    "name": func.get("name", "unknown"),
                    "start_line": func.get("location", {}).get("start_line", 0),
                    "end_line": func.get("location", {}).get("end_line", 0),
                }
            )

        metrics = analysis_data.get("complexity_metrics", {})

        with self.driver.session(database=self.db) as session:
            # Create file and analysis nodes
            session.run(
                """
                MERGE (f:SourceFile {path: $path})
                SET f.name = $name, f.language = $language
                MERGE (a:CodeAnalysis {id: $analysis_id})
                SET a.language = $language,
                    a.code_length = $code_length,
                    a.max_nesting_level = $max_nesting,
                    a.total_nodes = $total_nodes
                MERGE (f)-[:HAS_ANALYSIS]->(a)
                """,
                path=file_path,
                name=file_name,
                analysis_id=analysis_id,
                language=analysis_data.get("language", "unknown"),
                code_length=analysis_data.get("code_length", 0),
                max_nesting=metrics.get("max_nesting_level", 0),
                total_nodes=metrics.get("total_nodes", 0),
            )

            # Batch insert functions
            session.run(
                """
                MATCH (a:CodeAnalysis {id: $analysis_id})
                UNWIND $functions as f
                MERGE (func:Function {id: f.id})
                SET func.name = f.name,
                    func.start_line = f.start_line,
                    func.end_line = f.end_line
                MERGE (a)-[:HAS_FUNCTION]->(func)
                """,
                analysis_id=analysis_id,
                functions=functions,
            )

        return analysis_id


# Module-level singleton
_neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """Get or create the singleton Neo4j client."""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client

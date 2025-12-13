"""
Neo4j client for storing and querying AST/ASG data.

This module provides the integration with a Neo4j database to persist
code analysis results, enabling complex queries and visualizations.
"""

import hashlib
import os
from typing import Any, Dict, Optional

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
                print(f"Initialized Neo4j driver for {self.uri}")
            except Exception as e:
                print(f"Failed to initialize Neo4j driver: {e}")
                self.driver = None
        else:
            print("Neo4j driver not available. Install with 'uv add neo4j'")

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
        Store AST data in Neo4j.

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
                language=ast_data.get("language", "unknown"),
            )

            # Create AST node and link to file
            session.run(
                """
                MATCH (f:SourceFile {path: $path})
                MERGE (ast:AST {id: $ast_id})
                SET ast.language = $language
                MERGE (f)-[:HAS_AST]->(ast)
                """,
                path=file_path,
                ast_id=ast_id,
                language=ast_data.get("language", "unknown"),
            )

            # Store the AST nodes recursively
            if "ast" in ast_data:
                self._store_ast_node_recursive(session, ast_id, None, ast_data["ast"])

        return ast_id

    def _store_ast_node_recursive(
        self, session: Any, ast_id: str, parent_id: Optional[str], node: Dict[str, Any]
    ) -> None:
        """Recursively store AST nodes and their relationships."""
        # Generate a unique ID for this node
        node_id = f"{ast_id}_{node['type']}_{node.get('start_byte', 0)}_{node.get('end_byte', 0)}"

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
            type=node.get("type", "unknown"),
            text=node.get("text", "")[:1000],  # Truncate text to avoid massive strings
            start_byte=node.get("start_byte", 0),
            end_byte=node.get("end_byte", 0),
            start_line=node.get("start_point", {}).get("row", 0),
            start_col=node.get("start_point", {}).get("column", 0),
            end_line=node.get("end_point", {}).get("row", 0),
            end_col=node.get("end_point", {}).get("column", 0),
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
        for child in node.get("children", []):
            self._store_ast_node_recursive(session, ast_id, node_id, child)

    def store_asg(self, asg_data: Dict[str, Any], file_path: str) -> Optional[str]:
        """
        Store ASG data in Neo4j.

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
                language=asg_data.get("language", "unknown"),
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
                language=asg_data.get("language", "unknown"),
            )

            # Add nodes
            for node in asg_data.get("nodes", []):
                session.run(
                    """
                    MATCH (asg:ASG {id: $asg_id})
                    MERGE (n:ASGNode {id: $node_id})
                    SET n.type = $type,
                        n.text = $text,
                        n.start_byte = $start_byte,
                        n.end_byte = $end_byte
                    MERGE (asg)-[:CONTAINS]->(n)
                    """,
                    asg_id=asg_id,
                    node_id=node["id"],
                    type=node.get("type", "unknown"),
                    text=node.get("text", "")[:1000],
                    start_byte=node.get("start_byte", 0),
                    end_byte=node.get("end_byte", 0),
                )

            # Add edges
            for edge in asg_data.get("edges", []):
                session.run(
                    """
                    MATCH (s:ASGNode {id: $source_id})
                    MATCH (t:ASGNode {id: $target_id})
                    MERGE (s)-[r:EDGE {type: $edge_type}]->(t)
                    """,
                    source_id=edge["source"],
                    target_id=edge["target"],
                    edge_type=edge.get("type", "unknown"),
                )

        return asg_id

    def store_analysis(
        self, analysis_data: Dict[str, Any], file_path: str
    ) -> Optional[str]:
        """
        Store code analysis results in Neo4j.

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
                language=analysis_data.get("language", "unknown"),
            )

            # Create CodeAnalysis node
            metrics = analysis_data.get("complexity_metrics", {})
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
                language=analysis_data.get("language", "unknown"),
                code_length=analysis_data.get("code_length", 0),
                max_nesting=metrics.get("max_nesting_level", 0),
                total_nodes=metrics.get("total_nodes", 0),
            )

            # Add functions
            for func in analysis_data.get("functions", []):
                func_id = hashlib.md5(
                    f"{analysis_id}:func:{func['name']}:{func.get('location', {}).get('start_line', 0)}".encode(),
                    usedforsecurity=False,
                ).hexdigest()
                start_line = func.get("location", {}).get("start_line", 0)
                end_line = func.get("location", {}).get("end_line", 0)

                session.run(
                    """
                    MATCH (a:CodeAnalysis {id: $analysis_id})
                    MERGE (f:Function {id: $func_id})
                    SET f.name = $name,
                        f.start_line = $start_line,
                        f.end_line = $end_line
                    MERGE (a)-[:HAS_FUNCTION]->(f)
                    """,
                    analysis_id=analysis_id,
                    func_id=func_id,
                    name=func.get("name", "unknown"),
                    start_line=start_line,
                    end_line=end_line,
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

"""
Manages file-based analysis output for analyzed projects.

Saves analyses to structured folders with logical section breakdown:
- functions.json - All function definitions
- classes.json - All class definitions
- imports.json - All imports/dependencies
- structure.json - Overall code structure metrics
- metadata.json - Analysis timestamp and settings
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Output directory at project root
PROJECT_ROOT = Path(__file__).parent.parent
ANALYZED_PROJECTS_DIR = PROJECT_ROOT / "analyzed_projects"


class AnalysisOutputManager:
    """Manages saving and retrieving project analyses to the filesystem."""

    def __init__(self) -> None:
        ANALYZED_PROJECTS_DIR.mkdir(exist_ok=True)

    def create_analysis_folder(self, project_name: str) -> Path:
        """Create timestamped folder for a project analysis.

        Format: {project_name}_{YYYY-MM-DD}_{HH-MM}
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        safe_name = self._sanitize_name(project_name)
        folder_name = f"{safe_name}_{timestamp}"
        folder_path = ANALYZED_PROJECTS_DIR / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def save_analysis(
        self,
        project_name: str,
        ast_data: Dict[str, Any],
        *,
        asg_data: Optional[Dict[str, Any]] = None,
        structure_data: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None,
        source_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save analysis to logical section files.

        If output_path is provided, files are saved there.
        If source_filename is provided, it's used to create a subpath within output_path.
        """

        if output_path:
            folder = output_path
        else:
            folder = self.create_analysis_folder(project_name)

        if source_filename:
            # Create a per-file subfolder instead of just a prefix
            file_base = Path(source_filename).name.replace(".", "_")
            file_folder = folder / file_base
            file_folder.mkdir(parents=True, exist_ok=True)
            target_folder = file_folder
            file_prefix = ""
        else:
            target_folder = folder
            file_prefix = "analysis_"

        files_created: List[str] = []

        # Extract and save logical sections from structure data
        if structure_data and "error" not in structure_data:
            # Functions section
            functions = self._format_functions(structure_data.get("functions", []))
            fname = f"{file_prefix}functions.json"
            self._save_json(target_folder / fname, functions)
            files_created.append(
                str(Path(target_folder.name) / fname) if source_filename else fname
            )

            # Classes section
            classes = self._format_classes(structure_data.get("classes", []))
            cname = f"{file_prefix}classes.json"
            self._save_json(target_folder / cname, classes)
            files_created.append(
                str(Path(target_folder.name) / cname) if source_filename else cname
            )

            # Imports section
            imports = self._format_imports(structure_data.get("imports", []))
            iname = f"{file_prefix}imports.json"
            self._save_json(target_folder / iname, imports)
            files_created.append(
                str(Path(target_folder.name) / iname) if source_filename else iname
            )

            # Code metrics/structure overview
            metrics = {
                "language": structure_data.get("language", "unknown"),
                "code_length": structure_data.get("code_length", 0),
                "complexity_metrics": structure_data.get("complexity_metrics", {}),
                "summary": {
                    "total_functions": len(structure_data.get("functions", [])),
                    "total_classes": len(structure_data.get("classes", [])),
                    "total_imports": len(structure_data.get("imports", [])),
                },
            }
            sname = f"{file_prefix}structure.json"
            self._save_json(target_folder / sname, metrics)
            files_created.append(
                str(Path(target_folder.name) / sname) if source_filename else sname
            )

        # Save ASG if provided
        if asg_data and "error" not in asg_data:
            graph_summary = {
                "language": asg_data.get("language"),
                "total_nodes": len(asg_data.get("nodes", [])),
                "total_edges": len(asg_data.get("edges", [])),
                "root": asg_data.get("root"),
                "edge_types": self._count_edge_types(asg_data.get("edges", [])),
            }
            gname = f"{file_prefix}semantic_graph.json"
            self._save_json(target_folder / gname, graph_summary)
            files_created.append(
                str(Path(target_folder.name) / gname) if source_filename else gname
            )

        # Save full AST for reference
        if ast_data and "error" not in ast_data:
            ast_clean = {k: v for k, v in ast_data.items() if k != "tree_object"}
            aname = f"{file_prefix}full_ast.json"
            self._save_json(target_folder / aname, ast_clean)
            files_created.append(
                str(Path(target_folder.name) / aname) if source_filename else aname
            )

        # Save metadata (only if this is the primary or first analysis in the folder)
        metadata_path = folder / "metadata.json"
        if not metadata_path.exists():
            metadata = {
                "project_name": project_name,
                "analyzed_at": datetime.now().isoformat(),
                "language": ast_data.get("language") if ast_data else "unknown",
                "files_created": files_created,
                "output_folder": str(folder),
            }
            self._save_json(metadata_path, metadata)

        return {
            "folder": str(folder),
            "files_created": files_created,
        }

    def _format_functions(self, functions: List[Dict]) -> Dict[str, Any]:
        """Format functions into a readable structure."""
        return {
            "count": len(functions),
            "definitions": [
                {
                    "name": f.get("name", "unknown"),
                    "parameters": f.get("parameters", []),
                    "location": f.get("location", {}),
                }
                for f in functions
            ],
        }

    def _format_classes(self, classes: List[Dict]) -> Dict[str, Any]:
        """Format classes into a readable structure."""
        return {
            "count": len(classes),
            "definitions": [
                {
                    "name": c.get("name", "unknown"),
                    "location": c.get("location", {}),
                }
                for c in classes
            ],
        }

    def _format_imports(self, imports: List[Dict]) -> Dict[str, Any]:
        """Format imports into a readable structure."""
        return {
            "count": len(imports),
            "modules": [
                {
                    "module": i.get("module", "unknown"),
                    "line": i.get("line", 0),
                }
                for i in imports
            ],
        }

    def _count_edge_types(self, edges: List[Dict]) -> Dict[str, int]:
        """Count occurrences of each edge type in the ASG."""
        counts: Dict[str, int] = {}
        for edge in edges:
            edge_type = edge.get("type", "unknown")
            counts[edge_type] = counts.get(edge_type, 0) + 1
        return counts

    def _save_json(self, path: Path, data: Any) -> None:
        """Save data to JSON file with pretty formatting."""
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _sanitize_name(self, name: str) -> str:
        """Convert project name to safe filename."""
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# Module-level instance for convenience
_output_manager = AnalysisOutputManager()


def get_output_manager() -> AnalysisOutputManager:
    """Get the singleton output manager."""
    return _output_manager

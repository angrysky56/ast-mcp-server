"""
Enhanced CLI for AST MCP Server with transformation capabilities.

This module provides a rich command-line interface using typer and rich
for managing the AST MCP server with integrated ast-grep transformation tools.
"""

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

app = typer.Typer(
    name="ast-mcp-server",
    help="AST MCP Server with Code Analysis and Transformation",
    add_completion=False,
)

console = Console()


@app.command()
def serve(
    port: int = typer.Option(3000, "--port", "-p", help="Port to run the server on"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
) -> None:
    """Start the AST MCP Server with enhanced transformation capabilities."""
    console.print(
        Panel.fit(
            "[bold green]AST MCP Server[/bold green]\n"
            "[dim]Enhanced with ast-grep transformation tools[/dim]",
            border_style="green",
        )
    )

    if debug:
        console.print("[yellow]Debug mode enabled[/yellow]")

    # Import and run the server
    from ast_mcp_server.server import main

    main()


@app.command()
def build_parsers() -> None:
    """Build tree-sitter parsers for language support."""
    console.print("[blue]Building tree-sitter parsers...[/blue]")

    try:
        # trunk-ignore(bandit/B404)
        import subprocess

        # trunk-ignore(bandit/B603)
        result = subprocess.run(  # noqa: S603
            [sys.executable, "build_parsers.py"], capture_output=True, text=True
        )

        if result.returncode == 0:
            console.print("✅ [green]Parsers built successfully![/green]")
        else:
            console.print(f"❌ [red]Parser build failed:[/red] {result.stderr}")

    except Exception as e:
        console.print(f"❌ [red]Error building parsers:[/red] {e}")


@app.command()
def status() -> None:
    """Check the status of AST MCP Server components."""

    table = Table(title="AST MCP Server Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    # Check tree-sitter parsers
    try:
        from ast_mcp_server.tools import init_parsers

        if init_parsers():
            table.add_row(
                "Tree-sitter Parsers", "✅ Available", "All language parsers loaded"
            )
        else:
            table.add_row(
                "Tree-sitter Parsers", "⚠️ Limited", "Run 'build-parsers' command"
            )
    except Exception:
        table.add_row("Tree-sitter Parsers", "❌ Error", "Module import failed")

    # Check enhanced tools
    try:
        from ast_mcp_server.enhanced_tools import register_enhanced_tools  # noqa: F401

        table.add_row(
            "Enhanced Tools", "✅ Available", "Advanced AST/ASG analysis enabled"
        )
    except ImportError:
        table.add_row("Enhanced Tools", "⚠️ Missing", "Optional enhanced features")

    # Check transformation tools
    try:
        from ast_mcp_server.transformation_tools import (  # noqa: F401
            register_transformation_tools,
        )

        table.add_row(
            "Transformation Tools", "✅ Available", "ast-grep integration enabled"
        )
    except ImportError:
        table.add_row("Transformation Tools", "❌ Missing", "Install ast-grep-cli")

    console.print(table)


@app.command()
def test_pattern(
    pattern: str = typer.Argument(..., help="ast-grep pattern to test"),
    language: str = typer.Option("python", "--lang", "-l", help="Programming language"),
    code: Optional[str] = typer.Option(
        None, "--code", "-c", help="Test code (or use minimal test)"
    ),
) -> None:
    """Test an ast-grep pattern for validity."""

    try:
        from ast_mcp_server.transformation_tools import AstGrepTransformer

        transformer = AstGrepTransformer()

        if not code:
            # Use minimal test code
            test_codes = {
                "python": "def example(): pass\nif condition: example()",
                "javascript": "function example() {}\nif (condition) { example(); }",
                "typescript": "function example(): void {}\nif (condition) { example(); }",
            }
            code = test_codes.get(language, "// test code")

        console.print(f"[blue]Testing pattern:[/blue] {pattern}")
        console.print(f"[blue]Language:[/blue] {language}")
        console.print("[blue]Test code:[/blue]")
        console.print(Syntax(code, language, theme="github-dark"))

        matches = transformer.search_pattern(code, pattern, language)

        if matches:
            console.print(
                f"✅ [green]Pattern valid! Found {len(matches)} matches:[/green]"
            )
            for i, match in enumerate(matches, 1):
                console.print(f"  {i}. Line {match.start_line}: '{match.matched_text}'")
        else:
            console.print("⚠️ [yellow]Pattern valid but no matches found[/yellow]")

    except Exception as e:
        console.print(f"❌ [red]Pattern test failed:[/red] {e}")


@app.command()
def analyze_file(
    filename: str = typer.Option(..., "--filename", "-f", help="File to analyze"),
    project_name: str = typer.Option(
        ..., "--project-name", "-n", help="Name of the project output"
    ),
    language: Optional[str] = typer.Option(
        None, "--lang", "-l", help="Language override"
    ),
) -> None:
    """Analyze a single source file and save results to disk."""
    try:
        from ast_mcp_server.server import analyze_source_file as analyze_tool

        console.print(f"[blue]Analyzing {filename} as project {project_name}...[/blue]")

        result = analyze_tool(
            project_name=project_name, filename=filename, language=language
        )
        console.print("[green]Analysis complete![/green]")
        console.print(result)

    except Exception as e:
        console.print(f"❌ [red]Analysis failed:[/red] {e}")


@app.command()
def analyze_project(
    path: str = typer.Option(..., "--path", "-p", help="Directory to analyze"),
    project_name: str = typer.Option(
        ..., "--project-name", "-n", help="Name of the project output"
    ),
    sync: bool = typer.Option(True, "--sync/--no-sync", help="Sync to Graph DB"),
) -> None:
    """Recursively analyze a project directory."""
    try:
        from ast_mcp_server.server import analyze_project as analyze_tool

        console.print(f"[blue]Analyzing project at {path}...[/blue]")

        result = analyze_tool(
            project_path=path, project_name=project_name, sync_to_db=sync
        )

        console.print("[green]Project Analysis Complete![/green]")
        console.print(f"Processed: {result.get('processed_files', 0)}")
        console.print(f"Synced: {result.get('synced_files', 0)}")
        console.print(f"Failed: {result.get('failed_files', 0)}")

        if result.get("failures"):
            console.print("[red]Failures:[/red]")
            for f in result["failures"]:
                console.print(f"  {f['file']}: {f['error']}")

    except Exception as e:
        console.print(f"❌ [red]Project analysis failed:[/red] {e}")


if __name__ == "__main__":
    app()

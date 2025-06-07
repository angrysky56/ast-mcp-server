# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-06-06

### Added
- **pyproject.toml** configuration file for modern Python packaging
- **uv** package manager support for faster dependency management
- Entry point scripts: `ast-mcp-server` and `build-parsers`
- Development dependencies and tool configurations (black, isort, flake8, mypy)
- Testing infrastructure with pytest
- Documentation generation support with Sphinx
- Installation validation script (`validate_installation.py`)
- Enhanced .gitignore for uv and modern Python development

### Changed
- **BREAKING**: Minimum Python version increased from 3.9 to 3.10 (required by MCP)
- Converted from requirements.txt to pyproject.toml dependency management
- Updated README.md with comprehensive uv-based installation and usage instructions
- Updated Claude Desktop configuration to use new entry points
- Improved server.py and build_parsers.py with proper main() functions
- Enhanced error messages to reference uv commands

### Removed
- requirements.txt file (replaced by pyproject.toml)

### Technical Details
- Updated MCP dependency to latest version (1.9.3)
- Added comprehensive tool configuration for development workflow
- Improved package metadata and classifiers
- Added optional dependency groups for development, testing, and documentation

### Migration Guide
If you're upgrading from a previous version:

1. **Install uv** if you haven't already:
   ```bash
   pip install uv
   ```

2. **Remove old virtual environment and reinstall**:
   ```bash
   rm -rf .venv
   uv sync
   ```

3. **Update Claude Desktop configuration** to use new entry point:
   ```json
   {
     "mcpServers": {
       "AstAnalyzer": {
         "command": "uv",
         "args": [
           "--directory", "/path/to/ast-mcp-server",
           "run", "ast-mcp-server"
         ]
       }
     }
   }
   ```

4. **Rebuild parsers**:
   ```bash
   uv run build-parsers
   ```

## [0.1.0] - 2024-XX-XX

### Initial Release
- Basic AST parsing with tree-sitter
- ASG (Abstract Semantic Graph) generation
- Support for Python and JavaScript
- MCP server integration
- Claude Desktop compatibility
- Incremental parsing support
- Enhanced AST analysis tools

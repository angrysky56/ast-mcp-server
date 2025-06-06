#!/bin/bash
# Start the AST MCP server in development mode with the inspector

# Kill any running instances
pkill -f "mcp dev server.py" || true

# Current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start the server with the MCP inspector using uv
cd "$DIR" || exit
uv run -m mcp dev server.py

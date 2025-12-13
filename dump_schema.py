"""Dump all tool schemas from the MCP server."""

import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())


async def main() -> None:
    from ast_mcp_server.server import mcp

    print("--- All Tool Schemas ---\n")

    tools = await mcp.list_tools()
    for tool in tools:
        print(f"• {tool.name}")
        print(f"  {tool.description}")
        print()


if __name__ == "__main__":
    asyncio.run(main())

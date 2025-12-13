"""Dump full tool schema to verify parameters are preserved."""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.getcwd())


async def main() -> None:
    from ast_mcp_server.server import mcp

    print("--- Sample Tool Schema (sync_file_to_graph) ---\n")

    tools = await mcp.list_tools()
    for tool in tools:
        if tool.name == "sync_file_to_graph":
            print(f"Name: {tool.name}")
            print(f"Description: {tool.description}")
            print("\nInput Schema:")
            print(json.dumps(tool.inputSchema, indent=2))
            break


if __name__ == "__main__":
    asyncio.run(main())

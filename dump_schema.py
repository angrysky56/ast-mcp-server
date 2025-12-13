import asyncio
import json
import os
import sys

# Ensure we can import modules from current directory
sys.path.append(os.getcwd())


# Import the server tools registration
async def main():
    try:
        from ast_mcp_server.server import mcp

        # We trigger the tool registration by importing server, which runs the registration code
        # (since it's at module level in server.py).

        print("--- Tool Schemas ---")

        # FastMCP uses decorators to register tools. We can iterate over them.
        # The tools are typically stored in mcp._tool_manager._tools, but let's check the public API or iterate list_tools() output.

        tools = await mcp.list_tools()
        for tool in tools:
            if tool.name in ["sync_file_to_graph", "query_neo4j_graph"]:
                print(f"\nTool: {tool.name}")
                print(json.dumps(tool.inputSchema, indent=2))

    except ImportError as e:
        print(f"ImportError: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())

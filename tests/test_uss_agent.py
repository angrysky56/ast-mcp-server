"""
Test the USS Agent with a simple query.
"""

import os
import sys

sys.path.append(os.getcwd())

from ast_mcp_server.uss_agent_tools import ask_uss_agent, uss_agent_status


def main() -> None:
    print("--- USS Agent Test ---")

    # 1. Check status
    print("\n[1] Agent Status:")
    status = uss_agent_status()
    print(f"  {status}")

    # 2. Ask a question
    print("\n[2] Asking: 'How many source files are indexed?'")
    result = ask_uss_agent("How many source files are indexed in the graph?")

    if "error" in result:
        print(f"  Error: {result['error']}")
    else:
        print(f"  Decision: {result.get('decision', {}).get('action')}")
        print(f"  Database: {result.get('decision', {}).get('database')}")
        if result.get("neo4j_result"):
            print(f"  Neo4j Count: {result['neo4j_result'].get('count')}")
        print(f"\n  Summary: {result.get('summary', 'No summary')[:500]}")


if __name__ == "__main__":
    main()

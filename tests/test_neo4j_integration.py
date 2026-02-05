
import glob
import os
import sys

# Ensure we can import modules
sys.path.append(os.getcwd())

from ast_mcp_server.neo4j_tools import query_neo4j_graph, sync_file_to_graph


def main():
    print("--- Testing Neo4j Integration ---")

    # Files to process (limit to a few key files to be fast)
    files_to_scan = [
        "ast_mcp_server/server.py",
        "ast_mcp_server/tools.py",
        "ast_mcp_server/neo4j_tools.py"
    ]

    # 1. Sync files
    print("\n[1] Syncing files to Neo4j...")
    for file_path in files_to_scan:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            print(f"Skipping {file_path} (not found)")
            continue

        print(f"  Syncing {file_path}...")
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                code = f.read()

            result = sync_file_to_graph(code, abs_path)
            if "error" in result:
                print(f"    Error: {result['error']}")
            else:
                print(f"    Success: {result['status']}")
                print(f"    Stored IDs: {result.get('stored', {})}")
        except Exception as e:
            print(f"    Exception: {e}")

    # 2. Query Graph
    print("\n[2] Verifying with Cypher queries...")

    queries = [
        ("Count Nodes", "MATCH (n) RETURN count(n) as count"),
        ("Source Files", "MATCH (f:SourceFile) RETURN f.name as name, f.path as path"),
        ("Functions", "MATCH (f:Function) RETURN f.name as name, f.start_line as line LIMIT 5"),
    ]

    for label, query in queries:
        print(f"  Running query: {label}")
        result = query_neo4j_graph(query)
        if "error" in result:
             print(f"    Error: {result['error']}")
        else:
             print(f"    Count: {result.get('count')}")
             if result.get('records'):
                 print(f"    Sample: {result['records'][:2]}")

if __name__ == "__main__":
    main()

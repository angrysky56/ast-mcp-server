#!/usr/bin/env python
"""
Example demonstrating how to use the code analysis tools.

This script shows how to parse code into AST and ASG, and perform code analysis
using the existing tools.
"""

import os
import sys
import json

# Add the parent directory to the path so we can import the tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ast_mcp_server.tools import (
    parse_code_to_ast,
    create_asg_from_ast,
    analyze_code_structure,
    init_parsers
)

# Example Python code to analyze
EXAMPLE_CODE = """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Calculate factorial of 5
result = factorial(5)
print(f"Factorial of 5 is {result}")
"""

def main():
    """Main function to demonstrate code analysis tools."""
    print("Code Analysis Example")
    print("-" * 50)

    # Check if parsers are initialized
    print("Initializing parsers...")
    if not init_parsers():
        print("Warning: Tree-sitter parsers not available.")
        print("This example will show what would be returned if parsers were available.")

    # Parse the code to AST
    print("\n1. Parsing code to AST")
    ast_result = parse_code_to_ast(EXAMPLE_CODE, language="python")

    if "error" in ast_result:
        print(f"Error parsing code: {ast_result['error']}")
        print("Showing example AST structure:")
        # Show an example of what the AST would look like
        example_ast = {
            "language": "python",
            "ast": {
                "type": "module",
                "start_byte": 0,
                "end_byte": len(EXAMPLE_CODE),
                "children": [
                    {"type": "function_definition", "text": "def factorial(n):..."},
                    {"type": "function_definition", "text": "def fibonacci(n):..."},
                    {"type": "expression_statement", "text": "result = factorial(5)"},
                    {"type": "expression_statement", "text": "print(f\"Factorial of 5 is {result}\")"}
                ]
            }
        }
        print(json.dumps(example_ast, indent=2))
    else:
        # Print a simplified version of the AST
        print("AST generated successfully:")
        print(f"Language: {ast_result['language']}")
        print(f"Root node type: {ast_result['ast']['type']}")
        print(f"Number of children: {len(ast_result['ast'].get('children', []))}")

    # Generate ASG from AST
    print("\n2. Generating ASG")
    if "error" in ast_result:
        print("Cannot generate ASG without a valid AST.")
        # Show an example of what the ASG would look like
        example_asg = {
            "language": "python",
            "nodes": [
                {"id": "module_0_340", "type": "module", "text": "..."},
                {"id": "function_definition_1_82", "type": "function_definition", "text": "def factorial(n):..."},
                {"id": "identifier_5_14", "type": "identifier", "text": "factorial"}
            ],
            "edges": [
                {"source": "module_0_340", "target": "function_definition_1_82", "type": "contains"},
                {"source": "function_definition_1_82", "target": "identifier_5_14", "type": "contains"},
                {"source": "identifier_243_252", "target": "function_definition_1_82", "type": "calls"}
            ]
        }
        print(json.dumps(example_asg, indent=2))
    else:
        asg_result = create_asg_from_ast(ast_result)
        print("ASG generated successfully:")
        print(f"Number of nodes: {len(asg_result['nodes'])}")
        print(f"Number of edges: {len(asg_result['edges'])}")
        # Print a few sample nodes and edges
        if asg_result['nodes']:
            print("\nSample nodes:")
            for node in asg_result['nodes'][:3]:
                print(f"  {node['id']} ({node['type']}): {node['text'][:30]}...")
        if asg_result['edges']:
            print("\nSample edges:")
            for edge in asg_result['edges'][:3]:
                print(f"  {edge['source']} --{edge['type']}--> {edge['target']}")

    # Analyze code structure
    print("\n3. Analyzing code structure")
    analysis_result = analyze_code_structure(EXAMPLE_CODE, language="python")

    if "error" in analysis_result:
        print(f"Error analyzing code: {analysis_result['error']}")
        # Show an example of what the analysis would look like
        example_analysis = {
            "language": "python",
            "code_length": len(EXAMPLE_CODE),
            "functions": [
                {
                    "name": "factorial",
                    "location": {"start_line": 2, "end_line": 6},
                    "parameters": ["n"]
                },
                {
                    "name": "fibonacci",
                    "location": {"start_line": 8, "end_line": 14},
                    "parameters": ["n"]
                }
            ],
            "classes": [],
            "imports": [],
            "complexity_metrics": {
                "max_nesting_level": 2,
                "total_nodes": 67
            }
        }
        print(json.dumps(example_analysis, indent=2))
    else:
        print("Code analysis completed successfully:")
        print(f"Language: {analysis_result['language']}")
        print(f"Code length: {analysis_result['code_length']} bytes")
        print(f"Number of functions: {len(analysis_result['functions'])}")
        print(f"Number of classes: {len(analysis_result['classes'])}")
        print(f"Max nesting level: {analysis_result['complexity_metrics']['max_nesting_level']}")

        if analysis_result['functions']:
            print("\nFunctions:")
            for func in analysis_result['functions']:
                params = ", ".join(func['parameters'])
                print(f"  {func['name']}({params}) at lines {func['location']['start_line']}-{func['location']['end_line']}")

    print("\nIn a real Neo4j integration, you could:")
    print("1. Store AST/ASG nodes and edges in Neo4j")
    print("2. Use Cypher queries to find code patterns")
    print("3. Identify complex functions, unused imports, etc.")
    print("4. Visualize code structure and dependencies")

if __name__ == "__main__":
    main()

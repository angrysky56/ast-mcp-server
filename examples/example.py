#!/usr/bin/env python
"""
Example demonstrating how to use the AST/ASG Analysis MCP server.

This script shows how the tools of the AST MCP server can be used
to analyze code structure and semantics.
"""


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
    """Main function to run the example."""
    print("AST/ASG Analysis MCP Server Example")
    print("-" * 50)

    # In a real scenario, you would use an MCP client to connect to the server
    # For demonstration, let's show what you'd expect to see from the tools

    print("Sample code to analyze:")
    print("-" * 50)
    print(EXAMPLE_CODE)
    print("-" * 50)

    # Mock parse_to_ast tool call
    print("\n1. Parsing code to AST")
    print("The parse_to_ast tool would return a hierarchical tree like:")
    print("""
    {
      "language": "python",
      "ast": {
        "type": "module",
        "start_byte": 0,
        "end_byte": 340,
        "children": [
          {
            "type": "function_definition",
            "start_byte": 1,
            "end_byte": 82,
            "children": [
              {"type": "def", "text": "def"},
              {"type": "identifier", "text": "factorial"},
              {"type": "parameters", "children": [
                {"type": "identifier", "text": "n"}
              ]},
              {"type": ":", "text": ":"},
              {"type": "block", "children": [
                {"type": "if_statement", "children": [
                  {"type": "if", "text": "if"},
                  {"type": "comparison_operator", "children": [
                    {"type": "identifier", "text": "n"},
                    {"type": "<=", "text": "<="},
                    {"type": "integer", "text": "1"}
                  ]},
                  {"type": ":", "text": ":"},
                  # ... more nodes ...
                ]}
              ]}
            ]
          },
          # ... more function definitions ...
        ]
      }
    }
    """)

    # Mock generate_asg tool call
    print("\n2. Generating ASG")
    print("The generate_asg tool would create a graph with nodes and edges:")
    print("""
    {
      "language": "python",
      "nodes": [
        {"id": "module_0_340", "type": "module", "text": "..."},
        {"id": "function_definition_1_82", "type": "function_definition", "text": "def factorial(n):..."},
        {"id": "identifier_5_14", "type": "identifier", "text": "factorial"},
        {"id": "identifier_15_16", "type": "identifier", "text": "n"},
        # ... more nodes ...
      ],
      "edges": [
        {"source": "module_0_340", "target": "function_definition_1_82", "type": "contains"},
        {"source": "function_definition_1_82", "target": "identifier_5_14", "type": "contains"},
        {"source": "identifier_243_252", "target": "function_definition_1_82", "type": "calls"},
        # ... more edges including semantic relationships like "calls" and "references" ...
      ]
    }
    """)

    # Mock analyze_code tool call
    print("\n3. Analyzing code structure")
    print("The analyze_code tool would extract high-level structure:")
    print("""
    {
      "language": "python",
      "code_length": 340,
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
    """)

    print("\nIn a real MCP client interaction, you would:")
    print("1. Connect to the AST MCP server")
    print("2. Use the tools to analyze your code")
    print("3. Get back structured information about code syntax and semantics")
    print("4. Use that information to improve code understanding and generation")

if __name__ == "__main__":
    main()

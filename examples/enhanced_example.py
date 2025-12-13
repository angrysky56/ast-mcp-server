#!/usr/bin/env python
"""
Example demonstrating the enhanced AST/ASG Analysis MCP server features.

This script shows how the improved tools can be used for better
code structure and semantic analysis, with proper scope handling
and performance optimizations.
"""


# Example Python code to analyze - a bit more complex to showcase scope handling
EXAMPLE_CODE = """
# Global variable
global_var = 42

def factorial(n):
    # Local variable in factorial scope
    result = 1
    for i in range(1, n + 1):
        # i is in the for loop scope
        result *= i
    # Use global variable
    print(f"Global value is {global_var}")
    return result

class MathUtils:
    # Class variable
    description = "Utility class for math operations"

    def __init__(self, name):
        # Instance variable
        self.name = name

    def compute_factorial(self, n):
        # Method calls global function
        return factorial(n)

    @staticmethod
    def is_prime(n):
        # Static method with its own scope
        if n <= 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

# Create instance and use methods
math_utils = MathUtils("My Math Utils")
result1 = math_utils.compute_factorial(5)
result2 = MathUtils.is_prime(7)

print(f"Factorial of 5 is {result1}")
print(f"Is 7 prime? {result2}")
"""

# Slightly modified version to demonstrate incremental parsing and diff
MODIFIED_CODE = """
# Global variable
global_var = 42

def factorial(n):
    # Local variable in factorial scope
    result = 1
    for i in range(1, n + 1):
        # i is in the for loop scope
        result *= i
    # Use global variable
    print(f"Global value is {global_var}")
    return result

class MathUtils:
    # Class variable
    description = "Enhanced utility class for math operations"  # Changed description

    def __init__(self, name):
        # Instance variable
        self.name = name
        self.created_at = "2025-04-25"  # Added new instance variable

    def compute_factorial(self, n):
        # Method calls global function
        return factorial(n)

    # Added a new method
    def compute_square(self, n):
        return n * n

    @staticmethod
    def is_prime(n):
        # Improved prime check algorithm
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

# Create instance and use methods
math_utils = MathUtils("My Enhanced Math Utils")  # Changed name
result1 = math_utils.compute_factorial(5)
result2 = MathUtils.is_prime(7)
result3 = math_utils.compute_square(4)  # Added call to new method

print(f"Factorial of 5 is {result1}")
print(f"Is 7 prime? {result2}")
print(f"Square of 4 is {result3}")  # Added print statement
"""


def main() -> None:
    """Main function to demonstrate the enhanced AST MCP server features."""
    print("Enhanced AST/ASG Analysis MCP Server Example")
    print("-" * 60)

    # In a real scenario, you would use an MCP client to connect to the server
    # For demonstration, let's show what you'd expect to see from the enhanced tools

    print("Sample code to analyze:")
    print("-" * 60)
    print(EXAMPLE_CODE[:300] + "...")  # Show just the beginning for brevity
    print("-" * 60)

    # 1. Enhanced ASG Generation
    print("\n1. Generating Enhanced ASG")
    print("The generate_enhanced_asg tool would create a more complete graph with:")
    print("  - Proper scope hierarchy tracking")
    print("  - Complete variable definition and reference edges")
    print("  - Control flow edges between code blocks")
    print("  - Data flow edges showing dependencies")

    print("\nExample scope hierarchy detected:")
    print(
        """
    global                        // Global scope
    |
    ├── function:factorial       // Function scope
    |   └── for:i                // Loop scope inside function
    |
    └── class:MathUtils          // Class scope
        ├── method:__init__      // Method scope
        ├── method:compute_factorial
        └── method:is_prime
            └── for:i            // Loop scope inside method
    """
    )

    print("\nExample semantic edges detected:")
    print(
        """
    - CONTAINS: module → global_var, factorial, MathUtils, ...
    - DEFINES: global scope → global_var
    - DEFINES: factorial → result, i
    - REFERENCES: factorial(print) → global_var
    - CALLS: math_utils.compute_factorial → factorial
    - CONTROL_FLOW: for → block (inside factorial)
    - DATA_FLOW: result *= i → result, i
    """
    )

    # 2. Incremental Parsing and AST Diff
    print("\n2. Incremental Parsing and AST Diff")
    print("When code is modified, parse_to_ast_incremental and diff_ast tools would:")
    print("  - Only parse the changed parts of the code (much faster for large files)")
    print("  - Return only the nodes that changed")
    print("  - Highlight the specific changes between versions")

    print("\nExample of changes detected:")
    print(
        """
    - MODIFIED: MathUtils.description = "Enhanced utility class for math operations"
    - ADDED: self.created_at = "2025-04-25" in MathUtils.__init__
    - ADDED: compute_square method to MathUtils class
    - MODIFIED: is_prime method implementation
    - MODIFIED: MathUtils instance name to "My Enhanced Math Utils"
    - ADDED: result3 = math_utils.compute_square(4)
    - ADDED: print(f"Square of 4 is {result3}")
    """
    )

    # 3. Finding nodes at specific positions
    print("\n3. Finding Nodes at Positions")
    print("The find_node_at_position tool would help locate specific elements:")
    print("  - Find what function or variable is at a cursor position")
    print("  - Identify the scope a particular code belongs to")
    print("  - Navigate and refactor code more precisely")

    print("\nExample results:")
    print(
        """
    - Position (3, 10): function_definition "factorial"
    - Position (22, 15): method_definition "compute_factorial"
    - Position (40, 20): call "math_utils.compute_factorial"
    """
    )

    print("\nIn a real MCP client interaction, you would:")
    print("1. Connect to the Enhanced AST MCP server")
    print("2. Use the enhanced tools for more accurate code analysis")
    print("3. Get the benefits of proper scope handling and performance optimizations")
    print(
        "4. Use the results to improve code understanding, refactoring, or generation"
    )
    print(
        "5. Experience much faster processing for large codebases with incremental updates"
    )


if __name__ == "__main__":
    main()

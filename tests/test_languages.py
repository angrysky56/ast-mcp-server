#!/usr/bin/env python
"""Test script for tree-sitter language modules."""

import tree_sitter_python
import tree_sitter_javascript
from tree_sitter import Language, Parser

# Test Python
print("Setting up Python language...")
python_language = Language(tree_sitter_python.language())
python_parser = Parser()
# Use the language property instead of set_language method
python_parser.language = python_language

python_code = b"""
def hello():
    print("Hello, world!")
"""

python_tree = python_parser.parse(python_code)
print("Python parsing successful!")
print(f"Root node type: {python_tree.root_node.type}")
# Check what attributes are available on nodes
print("Node attributes:")
print(dir(python_tree.root_node))
# Print the tree structure using the string representation
print(f"Node representation: {python_tree.root_node}")
# Print children
print("Children:")
for i, child in enumerate(python_tree.root_node.children):
    print(f"Child {i}: {child.type}")
print("-" * 50)

# Test JavaScript
print("Setting up JavaScript language...")
js_language = Language(tree_sitter_javascript.language())
js_parser = Parser()
# Use the language property instead of set_language method
js_parser.language = js_language

js_code = b"""
function hello() {
    console.log("Hello, world!");
}
"""

js_tree = js_parser.parse(js_code)
print("JavaScript parsing successful!")
print(f"Root node type: {js_tree.root_node.type}")
# Print children
print("Children:")
for i, child in enumerate(js_tree.root_node.children):
    print(f"Child {i}: {child.type}")

#!/usr/bin/env python
"""Test script for tree-sitter-languages."""

import tree_sitter_languages
from tree_sitter import Parser

# Test a simple Python code snippet
code = b"""
def hello():
    print("Hello, world!")
"""

# Get a parser for Python
parser = tree_sitter_languages.get_parser('python')
tree = parser.parse(code)
root = tree.root_node

print(f"Python root node type: {root.type}")
print(f"S-expression: {root.sexp()}")

# Alternative way using get_language and Parser
python_lang = tree_sitter_languages.get_language('python')
custom_parser = Parser()
custom_parser.set_language(python_lang)

tree2 = custom_parser.parse(code)
print(f"Tree2 root node type: {tree2.root_node.type}")

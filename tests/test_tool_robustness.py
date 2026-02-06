import os

import pytest

from ast_mcp_server.tools import is_placeholder, parse_code_to_ast
from ast_mcp_server.transformation_tools import AstGrepTransformer


def test_is_placeholder():
    assert is_placeholder("") is True
    assert is_placeholder("# Read from file") is True
    assert is_placeholder("def foo(): pass") is False
    assert is_placeholder("import os") is False
    assert is_placeholder("x = 1") is True  # Short, no import/def


def test_parse_code_to_ast_fallback():
    # Create a temporary file
    test_file = "test_robustness.py"
    content = "def hello_world():\n    print('Hello World')\n"
    with open(test_file, "w") as f:
        f.write(content)

    try:
        # Call with placeholder code
        result = parse_code_to_ast(code="# Placeholder", filename=test_file)
        assert "error" not in result
        assert result["language"] == "python"
        # Check if it actually read the file by looking for 'hello_world' in the AST text
        assert "hello_world" in result["ast"]["text"]
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


def test_search_pattern_fallback():
    transformer = AstGrepTransformer()
    test_file = "test_search.py"
    content = "async def recall(query, limit=5):\n    pass\n"
    with open(test_file, "w") as f:
        f.write(content)

    try:
        # Call with placeholder code
        matches = transformer.search_pattern(
            code="# Placeholder", pattern="async def $FUNC($$$ARGS)", filename=test_file
        )
        assert len(matches) > 0
        assert (
            matches[0]
            .matched_text.strip()
            .startswith("async def recall(query, limit=5):")
        )
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


if __name__ == "__main__":
    pytest.main([__file__])

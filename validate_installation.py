#!/usr/bin/env python
"""
Validation script to test the AST MCP Server installation and functionality.
"""

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import ast_mcp_server
        print("✓ ast_mcp_server package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ast_mcp_server: {e}")
        return False
    
    try:
        from ast_mcp_server.tools import parse_code_to_ast, create_asg_from_ast
        print("✓ Core tools imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core tools: {e}")
        return False
    
    try:
        from ast_mcp_server.enhanced_tools import parse_code_to_ast_incremental
        print("✓ Enhanced tools imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import enhanced tools: {e}")
        return False
    
    try:
        from mcp.server.fastmcp import FastMCP
        print("✓ MCP framework imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MCP framework: {e}")
        return False
        
    return True

def test_basic_functionality():
    """Test basic AST parsing functionality."""
    print("\nTesting basic functionality...")
    
    from ast_mcp_server.tools import parse_code_to_ast
    
    # Test Python code parsing
    python_code = "def hello(): print('world')"
    result = parse_code_to_ast(python_code, "python")
    
    if "error" in result:
        print(f"✗ Python parsing failed: {result['error']}")
        return False
    else:
        print("✓ Python code parsed successfully")
    
    # Test JavaScript code parsing
    js_code = "function hello() { console.log('world'); }"
    result = parse_code_to_ast(js_code, "javascript")
    
    if "error" in result:
        print(f"✗ JavaScript parsing failed: {result['error']}")
        return False
    else:
        print("✓ JavaScript code parsed successfully")
    
    return True

def test_asg_generation():
    """Test ASG generation functionality."""
    print("\nTesting ASG generation...")
    
    from ast_mcp_server.tools import parse_code_to_ast, create_asg_from_ast
    
    python_code = """
def add(a, b):
    return a + b

result = add(1, 2)
"""
    
    # Parse to AST first
    ast_result = parse_code_to_ast(python_code, "python")
    if "error" in ast_result:
        print(f"✗ AST generation failed: {ast_result['error']}")
        return False
    
    # Generate ASG
    asg_result = create_asg_from_ast(ast_result)
    if "error" in asg_result:
        print(f"✗ ASG generation failed: {asg_result['error']}")
        return False
    
    print("✓ ASG generated successfully")
    print(f"  Nodes: {len(asg_result.get('nodes', []))}")
    print(f"  Edges: {len(asg_result.get('edges', []))}")
    
    return True

def test_enhanced_features():
    """Test enhanced functionality."""
    print("\nTesting enhanced features...")
    
    try:
        from ast_mcp_server.enhanced_tools import parse_code_to_ast_incremental, create_enhanced_asg_from_ast
        
        python_code = """
class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

calc = Calculator()
result = calc.add(5, 3)
"""
        
        # Test incremental parsing
        ast_result = parse_code_to_ast_incremental(python_code, "python")
        if "error" in ast_result:
            print(f"✗ Incremental parsing failed: {ast_result['error']}")
            return False
        
        print("✓ Incremental parsing works")
        
        # Test enhanced ASG
        enhanced_asg = create_enhanced_asg_from_ast(ast_result)
        if "error" in enhanced_asg:
            print(f"✗ Enhanced ASG generation failed: {enhanced_asg['error']}")
            return False
        
        print("✓ Enhanced ASG generation works")
        print(f"  Enhanced nodes: {len(enhanced_asg.get('nodes', []))}")
        print(f"  Enhanced edges: {len(enhanced_asg.get('edges', []))}")
        
        return True
        
    except ImportError:
        print("⚠ Enhanced features not available (enhanced_tools.py not found)")
        return True  # This is not a failure

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("AST MCP Server Installation Validation")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    tests = [
        test_imports,
        test_basic_functionality,
        test_asg_generation,
        test_enhanced_features,
    ]
    
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! AST MCP Server is ready to use.")
        print("\nNext steps:")
        print("1. Configure Claude Desktop with the server path")
        print("2. Restart Claude Desktop")
        print("3. Start using AST analysis tools in your conversations")
    else:
        print("❌ Some tests failed. Please check the installation.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

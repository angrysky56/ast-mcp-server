#!/usr/bin/env python
"""
Script to setup tree-sitter language parsers using individual language modules.
This script prepares the language parsers for use in the AST MCP server.
"""

import os
import importlib
import importlib.util
from tree_sitter import Language, Parser

# Define the path to store parser-related files
PARSERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ast_mcp_server/parsers")
os.makedirs(PARSERS_PATH, exist_ok=True)

# Define the language modules to use
LANGUAGE_MODULES = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "java": "tree_sitter_java",
}

def install_missing_modules():
    """Install any missing tree-sitter language modules."""
    missing_modules = []
    for language, module_name in LANGUAGE_MODULES.items():
        try:
            importlib.import_module(module_name)
            print(f"✓ {module_name} is already installed")
        except ImportError:
            missing_modules.append(module_name)

    if missing_modules:
        print(f"The following modules need to be installed: {', '.join(missing_modules)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_modules)}")
        return False

    return True

def setup_languages():
    """Setup tree-sitter languages and test them."""
    languages = {}

    for lang_name, module_name in LANGUAGE_MODULES.items():
        try:
            # Import the language module
            module = importlib.import_module(module_name)

            # Get the language object
            lang = Language(module.language())
            languages[lang_name] = lang
            print(f"Successfully loaded {lang_name} language")
        except Exception as e:
            print(f"Error loading {lang_name} language: {e}")

    return languages

def test_parsers(languages):
    """Test that the parsers work correctly."""
    success = True

    for lang_name, language in languages.items():
        try:
            # Create a parser for this language
            parser = Parser()
            parser.language = language

            # Create a simple test snippet for each language
            if lang_name == "python":
                test_code = b"def hello(): print('world')"
            elif lang_name == "javascript":
                test_code = b"function hello() { console.log('world'); }"
            elif lang_name == "typescript":
                test_code = b"function hello(): string { console.log('world'); return 'hello'; }"
            elif lang_name == "go":
                test_code = b"func main() { fmt.Println(\"Hello World\") }"
            elif lang_name == "rust":
                test_code = b"fn main() { println!(\"Hello World\"); }"
            elif lang_name in ["c", "cpp"]:
                test_code = b"int main() { printf(\"Hello World\\n\"); return 0; }"
            elif lang_name == "java":
                test_code = b"class Main { public static void main(String[] args) { System.out.println(\"Hello World\"); } }"
            else:
                test_code = b"// Test code for " + lang_name.encode()

            # Parse the code and get the AST
            tree = parser.parse(test_code)
            root_node = tree.root_node

            print(f"Successfully tested {lang_name} parser")
            print(f"  Root node type: {root_node.type}")
            print(f"  Tree structure: {root_node}")
            print("-" * 50)

        except Exception as e:
            print(f"Error testing {lang_name} parser: {e}")
            success = False

    return success

def write_parser_info(languages):
    """Write parser info to a file that can be loaded by the server."""
    # Create a file to indicate parsers are available and list supported languages
    with open(os.path.join(PARSERS_PATH, "parsers_available.txt"), "w") as f:
        f.write("Tree-sitter language parsers are available.\n")
        f.write("LANGUAGES: " + ", ".join(languages.keys()))

def main():
    """Main entry point for building parsers."""
    try:
        print("Checking if required language modules are installed...")
        if not install_missing_modules():
            print("Please install the missing language modules and try again.")
            exit(1)

        print("Setting up tree-sitter languages...")
        languages = setup_languages()

        if not languages:
            print("No languages were loaded successfully.")
            exit(1)

        print("Testing parsers...")
        if test_parsers(languages):
            print("All parsers tested successfully!")
            write_parser_info(languages)
            print("Parser setup completed!")
        else:
            print("Some parsers failed testing.")
            exit(1)
    except Exception as e:
        print(f"Error setting up parsers: {e}")
        exit(1)


if __name__ == "__main__":
    main()

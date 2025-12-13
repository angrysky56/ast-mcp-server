"""
AST Code Transformation Tools for the MCP server.

This module integrates ast-grep for structural code transformation capabilities,
complementing the existing AST/ASG analysis tools with pattern-based search and replace.
"""

import json
import logging
import shutil

# Security Note: We import subprocess to run the 'ast-grep' CLI tool.
# We ensure safety by:
# 1. Not using shell=True (avoiding shell injection)
# 2. Passing arguments as a list
# 3. Using resolved absolute paths for the executable
# trunk-ignore(bandit/B404)
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import our existing tools for language detection and validation
from .tools import LANGUAGE_MAP, detect_language

logger = logging.getLogger(__name__)


@dataclass
class TransformationResult:
    """Result of a code transformation operation."""

    original_code: str
    transformed_code: str
    language: str
    pattern: str
    replacement: Optional[str] = None
    matches_found: int = 0
    changes_applied: int = 0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class PatternMatch:
    """A single pattern match in code."""

    start_line: int
    end_line: int
    start_column: int
    end_column: int
    matched_text: str
    file_path: Optional[str] = None


class AstGrepTransformer:
    """
    Elegant wrapper for ast-grep functionality with enhanced error handling and UX.

    Follows KISS principle by providing simple, focused methods for common operations.
    """

    def __init__(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ast_mcp_"))
        # Resolve full path to avoid partial executable path (bandit B607)
        self.ast_grep_path = shutil.which("ast-grep") or "ast-grep"
        self._check_ast_grep_available()

    def _check_ast_grep_available(self) -> bool:
        """Check if ast-grep is available in the system."""
        try:
            # Safe to run: fixed command, no untrusted input.
            # trunk-ignore(bandit/B603)
            result = subprocess.run(  # noqa: S603
                [self.ast_grep_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning(
                "ast-grep not found in PATH. Install with: pip install ast-grep-cli"
            )
            return False

    def search_pattern(
        self,
        code: str,
        pattern: str,
        language: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> List[PatternMatch]:
        """
        Search for structural patterns in code using ast-grep.

        Args:
            code: Source code to search
            pattern: ast-grep pattern (e.g., '$PROP && $PROP()')
            language: Programming language (auto-detected if not provided)
            filename: Optional filename for context

        Returns:
            List of pattern matches found
        """
        if not language:
            language = detect_language(code, filename)

        # Normalize language for ast-grep
        ast_grep_lang = self._normalize_language_for_ast_grep(language)

        # Create temporary file
        temp_file = self.temp_dir / f"search.{self._get_file_extension(language)}"
        temp_file.write_text(code, encoding="utf-8")

        try:
            # Run ast-grep search
            cmd = [
                self.ast_grep_path,
                "--pattern",
                pattern,
                "--lang",
                ast_grep_lang,
                "--json",
                str(temp_file),
            ]

            # Safe to run: cmd is a list of arguments, avoiding shell injection.
            # trunk-ignore(bandit/B603)
            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                logger.error(f"ast-grep search failed: {result.stderr}")
                return []

            # Parse JSON output
            matches = []
            if result.stdout.strip():
                search_results = json.loads(result.stdout)
                for match_data in search_results:
                    match = PatternMatch(
                        start_line=match_data.get("range", {})
                        .get("start", {})
                        .get("line", 0),
                        end_line=match_data.get("range", {})
                        .get("end", {})
                        .get("line", 0),
                        start_column=match_data.get("range", {})
                        .get("start", {})
                        .get("column", 0),
                        end_column=match_data.get("range", {})
                        .get("end", {})
                        .get("column", 0),
                        matched_text=match_data.get("text", ""),
                        file_path=str(temp_file),
                    )
                    matches.append(match)

            return matches

        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error during pattern search: {e}")
            return []
        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()

    def replace_pattern(
        self,
        code: str,
        pattern: str,
        replacement: str,
        language: Optional[str] = None,
        filename: Optional[str] = None,
        interactive: bool = False,
    ) -> TransformationResult:
        """
        Replace structural patterns in code using ast-grep.

        Args:
            code: Source code to transform
            pattern: ast-grep pattern to match
            replacement: Replacement pattern
            language: Programming language (auto-detected if not provided)
            filename: Optional filename for context
            interactive: Whether to prompt for each replacement

        Returns:
            TransformationResult with details of the operation
        """
        if not language:
            language = detect_language(code, filename)

        # First, search for matches to count them
        matches = self.search_pattern(code, pattern, language, filename)

        if not matches:
            return TransformationResult(
                original_code=code,
                transformed_code=code,
                language=language,
                pattern=pattern,
                replacement=replacement,
                matches_found=0,
                changes_applied=0,
                success=True,
            )

        # Normalize language for ast-grep
        ast_grep_lang = self._normalize_language_for_ast_grep(language)

        # Create temporary file
        temp_file = self.temp_dir / f"transform.{self._get_file_extension(language)}"
        temp_file.write_text(code, encoding="utf-8")

        try:
            # Build ast-grep command
            cmd = [
                self.ast_grep_path,
                "--pattern",
                pattern,
                "--rewrite",
                replacement,
                "--lang",
                ast_grep_lang,
                str(temp_file),
            ]

            if interactive:
                cmd.append("--interactive")
            else:
                cmd.append("--update-all")

            # Safe to run: cmd is a list of arguments, avoiding shell injection.
            # trunk-ignore(bandit/B603)
            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                return TransformationResult(
                    original_code=code,
                    transformed_code=code,
                    language=language,
                    pattern=pattern,
                    replacement=replacement,
                    matches_found=len(matches),
                    changes_applied=0,
                    success=False,
                    error_message=result.stderr or "Unknown ast-grep error",
                )

            # Read the transformed code
            transformed_code = temp_file.read_text(encoding="utf-8")
            changes_applied = len(matches) if transformed_code != code else 0

            return TransformationResult(
                original_code=code,
                transformed_code=transformed_code,
                language=language,
                pattern=pattern,
                replacement=replacement,
                matches_found=len(matches),
                changes_applied=changes_applied,
                success=True,
            )

        except subprocess.TimeoutExpired:
            return TransformationResult(
                original_code=code,
                transformed_code=code,
                language=language,
                pattern=pattern,
                replacement=replacement,
                matches_found=len(matches),
                changes_applied=0,
                success=False,
                error_message="ast-grep operation timed out",
            )
        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()

    def validate_pattern(self, pattern: str, language: str) -> Dict[str, Any]:
        """
        Validate an ast-grep pattern for syntax correctness.

        Args:
            pattern: Pattern to validate
            language: Target language

        Returns:
            Dictionary with validation results
        """
        # Create minimal test code for the language
        test_code = self._get_minimal_test_code(language)

        # Try to search with the pattern
        try:
            matches = self.search_pattern(test_code, pattern, language)
            return {
                "valid": True,
                "pattern": pattern,
                "language": language,
                "test_matches": len(matches),
            }
        except Exception as e:
            return {
                "valid": False,
                "pattern": pattern,
                "language": language,
                "error": str(e),
            }

    def get_supported_languages(self) -> List[str]:
        """Get list of languages supported by both tree-sitter and ast-grep."""
        # Languages that both our tree-sitter setup and ast-grep support
        return list(LANGUAGE_MAP.keys())

    def _normalize_language_for_ast_grep(self, language: str) -> str:
        """Convert our language identifiers to ast-grep format."""
        # ast-grep language mapping
        ast_grep_map = {
            "python": "python",
            "javascript": "javascript",
            "typescript": "typescript",
            "tsx": "tsx",
            "go": "go",
            "rust": "rust",
            "c": "c",
            "cpp": "cpp",
            "java": "java",
        }

        normalized = LANGUAGE_MAP.get(language.lower(), language.lower())
        return ast_grep_map.get(normalized, normalized)

    def _get_file_extension(self, language: str) -> str:
        """Get appropriate file extension for language."""
        ext_map = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "tsx": "tsx",
            "go": "go",
            "rust": "rs",
            "c": "c",
            "cpp": "cpp",
            "java": "java",
        }
        normalized = LANGUAGE_MAP.get(language.lower(), language.lower())
        return ext_map.get(normalized, "txt")

    def _get_minimal_test_code(self, language: str) -> str:
        """Get minimal valid code for testing patterns."""
        test_code = {
            "python": "def test(): pass",
            "javascript": "function test() {}",
            "typescript": "function test(): void {}",
            "tsx": "const Test = () => <div></div>;",
            "go": "package main\nfunc test() {}",
            "rust": "fn test() {}",
            "c": "int test() { return 0; }",
            "cpp": "int test() { return 0; }",
            "java": "public class Test { public void test() {} }",
        }
        normalized = LANGUAGE_MAP.get(language.lower(), language.lower())
        return test_code.get(normalized, "// test code")

    def __del__(self) -> None:
        """Cleanup temporary directory."""
        try:
            import shutil

            if hasattr(self, "temp_dir") and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except (OSError, ImportError):
            pass  # Silent cleanup


def register_transformation_tools(mcp_server: Any) -> None:
    """Register ast-grep transformation tools with the MCP server."""

    transformer = AstGrepTransformer()

    @mcp_server.tool()
    def search_code_patterns(
        code: str,
        pattern: str,
        language: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for structural patterns in code using ast-grep.

        This tool performs structural search based on the Abstract Syntax Tree,
        allowing you to find code patterns that match the semantic structure
        rather than just text patterns.

        Args:
            code: Source code to search in
            pattern: ast-grep pattern (e.g., '$PROP && $PROP()' for property checks)
            language: Programming language (auto-detected if not provided)
            filename: Optional filename for language detection

        Returns:
            Dictionary containing search results with match locations and details
        """
        try:
            matches = transformer.search_pattern(code, pattern, language, filename)

            return {
                "success": True,
                "pattern": pattern,
                "language": language or detect_language(code, filename),
                "matches_found": len(matches),
                "matches": [asdict(match) for match in matches],
            }

        except Exception as e:
            return {"success": False, "error": str(e), "pattern": pattern}

    @mcp_server.tool()
    def transform_code_patterns(
        code: str,
        pattern: str,
        replacement: str,
        language: Optional[str] = None,
        filename: Optional[str] = None,
        preview_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Transform code by replacing structural patterns using ast-grep.

        This tool performs structural find-and-replace operations based on
        the Abstract Syntax Tree, enabling safe code refactoring and modernization.

        Args:
            code: Source code to transform
            pattern: ast-grep pattern to match (e.g., '$PROP && $PROP()')
            replacement: Replacement pattern (e.g., '$PROP?.()')
            language: Programming language (auto-detected if not provided)
            filename: Optional filename for language detection
            preview_only: If True, only show what would be changed without applying

        Returns:
            Dictionary containing transformation results and statistics
        """
        try:
            if preview_only:
                # Just search to show what would be transformed
                matches = transformer.search_pattern(code, pattern, language, filename)
                return {
                    "success": True,
                    "preview_mode": True,
                    "pattern": pattern,
                    "replacement": replacement,
                    "language": language or detect_language(code, filename),
                    "matches_found": len(matches),
                    "matches": [asdict(match) for match in matches],
                    "original_code": code,
                    "transformed_code": None,
                }

            # Perform actual transformation
            result = transformer.replace_pattern(
                code, pattern, replacement, language, filename, interactive=False
            )

            return {
                "success": result.success,
                "pattern": result.pattern,
                "replacement": result.replacement,
                "language": result.language,
                "matches_found": result.matches_found,
                "changes_applied": result.changes_applied,
                "original_code": result.original_code,
                "transformed_code": result.transformed_code,
                "error_message": result.error_message,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "pattern": pattern,
                "replacement": replacement,
            }

    @mcp_server.tool()
    def validate_ast_pattern(pattern: str, language: str) -> Dict[str, Any]:
        """
        Validate an ast-grep pattern for syntax correctness.

        This tool checks if a pattern is valid for the specified language
        before using it in search or transformation operations.

        Args:
            pattern: ast-grep pattern to validate
            language: Target programming language

        Returns:
            Dictionary with validation results and any error messages
        """
        try:
            result = transformer.validate_pattern(pattern, language)
            return result

        except Exception as e:
            return {
                "valid": False,
                "pattern": pattern,
                "language": language,
                "error": str(e),
            }

    @mcp_server.tool()
    def list_transformation_examples() -> Dict[str, Any]:
        """
        Get common ast-grep pattern examples for different languages.

        This tool provides a curated list of useful transformation patterns
        for code modernization, refactoring, and maintenance tasks.

        Returns:
            Dictionary containing example patterns organized by language and use case
        """
        examples = {
            "javascript": {
                "modernization": [
                    {
                        "name": "Optional Chaining",
                        "pattern": "$PROP && $PROP()",
                        "replacement": "$PROP?.()",
                        "description": "Convert defensive property checks to optional chaining",
                    },
                    {
                        "name": "Arrow Functions",
                        "pattern": "function($ARGS) { return $BODY }",
                        "replacement": "($ARGS) => $BODY",
                        "description": "Convert simple functions to arrow functions",
                    },
                ],
                "refactoring": [
                    {
                        "name": "Const Assertions",
                        "pattern": "var $VAR = $VALUE",
                        "replacement": "const $VAR = $VALUE",
                        "description": "Replace var with const for immutable values",
                    }
                ],
            },
            "python": {
                "modernization": [
                    {
                        "name": "F-strings",
                        "pattern": "'%s' % $VAR",
                        "replacement": "f'{$VAR}'",
                        "description": "Convert string formatting to f-strings",
                    },
                    {
                        "name": "Type Hints",
                        "pattern": "def $FUNC($ARGS):",
                        "replacement": "def $FUNC($ARGS) -> None:",
                        "description": "Add return type hints to functions",
                    },
                ]
            },
            "general": {
                "search_patterns": [
                    {
                        "name": "Function Calls",
                        "pattern": "$FUNC($$$ARGS)",
                        "description": "Match any function call",
                    },
                    {
                        "name": "Class Definitions",
                        "pattern": "class $NAME { $$$BODY }",
                        "description": "Match class definitions",
                    },
                ]
            },
        }

        return {
            "success": True,
            "examples": examples,
            "supported_languages": transformer.get_supported_languages(),
        }


def create_transformation_report(result: TransformationResult) -> str:
    """Create a rich formatted report of transformation results."""

    if not result.success:
        return f"❌ Transformation failed: {result.error_message}"

    if result.matches_found == 0:
        return f"ℹ️  No matches found for pattern: {result.pattern}"

    report = []
    report.append("✅ Pattern transformation completed")
    report.append(f"🔍 Pattern: {result.pattern}")
    if result.replacement:
        report.append(f"🔄 Replacement: {result.replacement}")
    report.append(f"📊 Matches found: {result.matches_found}")
    report.append(f"✏️  Changes applied: {result.changes_applied}")
    report.append(f"🗣️  Language: {result.language}")

    return "\n".join(report)

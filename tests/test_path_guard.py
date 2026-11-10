"""Regression tests for GitHub issue #2 (path traversal / arbitrary file read).

Each test builds two temporary folders: ``allowed`` (on the allow-list) and
``outside`` (not on it), then checks that every tool entry point that accepts
a path reads from the first and refuses the second.
"""

import os
from pathlib import Path

import pytest

from ast_mcp_server import server
from ast_mcp_server.enhanced_tools import parse_code_to_ast_incremental
from ast_mcp_server.path_guard import (
    PathNotAllowedError,
    get_allowed_roots,
    resolve_allowed_path,
    safe_cache_key,
)
from ast_mcp_server.resources import CACHE_DIR, get_cache_path
from ast_mcp_server.tools import analyze_code_structure, parse_code_to_ast
from ast_mcp_server.transformation_tools import AstGrepTransformer

SECRET = "def top_secret_marker():\n    import os\n    return 'do-not-leak'\n"
PUBLIC = "def public_marker():\n    import os\n    return 'fine to read'\n"


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Create an allowed folder and an outside folder, and set the allow-list."""
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "ok.py").write_text(PUBLIC)
    (outside / "secret.py").write_text(SECRET)
    monkeypatch.setenv("AST_MCP_ALLOWED_ROOTS", str(allowed))
    monkeypatch.chdir(allowed)
    server.AST_CACHE.clear()
    return {"allowed": allowed, "outside": outside}


# --- the guard itself -------------------------------------------------------


def test_allows_file_inside_root(sandbox: dict) -> None:
    assert resolve_allowed_path("ok.py") == sandbox["allowed"] / "ok.py"


def test_rejects_absolute_path_outside_root(sandbox: dict) -> None:
    with pytest.raises(PathNotAllowedError):
        resolve_allowed_path(str(sandbox["outside"] / "secret.py"))


def test_rejects_dotdot_traversal(sandbox: dict) -> None:
    with pytest.raises(PathNotAllowedError):
        resolve_allowed_path("../outside/secret.py")


def test_rejects_sibling_folder_with_same_prefix(sandbox: dict) -> None:
    # "allowed_evil" starts with the same characters as "allowed"
    evil = sandbox["allowed"].parent / "allowed_evil"
    evil.mkdir()
    (evil / "x.py").write_text(SECRET)
    with pytest.raises(PathNotAllowedError):
        resolve_allowed_path(str(evil / "x.py"))


def test_rejects_symlink_pointing_outside(sandbox: dict) -> None:
    link = sandbox["allowed"] / "link.py"
    link.symlink_to(sandbox["outside"] / "secret.py")
    with pytest.raises(PathNotAllowedError):
        resolve_allowed_path(str(link))


@pytest.mark.parametrize("name", [".env", ".env.local", "server.pem", "id_rsa"])
def test_rejects_credential_files_inside_root(sandbox: dict, name: str) -> None:
    (sandbox["allowed"] / name).write_text("API_KEY=abc")
    with pytest.raises(PathNotAllowedError):
        resolve_allowed_path(name)


def test_rejects_ssh_folder_inside_root(sandbox: dict) -> None:
    ssh = sandbox["allowed"] / ".ssh"
    ssh.mkdir()
    (ssh / "config").write_text("Host x")
    with pytest.raises(PathNotAllowedError):
        resolve_allowed_path(str(ssh / "config"))


@pytest.mark.parametrize("bad", ["", "   ", "a\x00b.py"])
def test_rejects_empty_and_malformed(sandbox: dict, bad: str) -> None:
    with pytest.raises(PathNotAllowedError):
        resolve_allowed_path(bad)


def test_default_root_is_cwd(sandbox: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AST_MCP_ALLOWED_ROOTS")
    assert get_allowed_roots() == [sandbox["allowed"]]


def test_multiple_roots(sandbox: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    both = os.pathsep.join([str(sandbox["allowed"]), str(sandbox["outside"])])
    monkeypatch.setenv("AST_MCP_ALLOWED_ROOTS", both)
    assert resolve_allowed_path(str(sandbox["outside"] / "secret.py")).is_file()


# --- tool entry points named in the issue (and the ones it missed) ----------


def test_parse_reads_allowed_file(sandbox: dict) -> None:
    result = parse_code_to_ast(filename="ok.py")
    assert "error" not in result
    assert "public_marker" in result["ast"]["text"]


@pytest.mark.parametrize("how", ["absolute", "dotdot"])
def test_parse_refuses_outside_file(sandbox: dict, how: str) -> None:
    target = (
        str(sandbox["outside"] / "secret.py")
        if how == "absolute"
        else "../outside/secret.py"
    )
    result = parse_code_to_ast(filename=target)
    assert "error" in result
    assert "top_secret_marker" not in str(result)


def test_parse_placeholder_does_not_read_outside_file(sandbox: dict) -> None:
    result = parse_code_to_ast(
        code="# Read from file", filename=str(sandbox["outside"] / "secret.py")
    )
    assert "top_secret_marker" not in str(result)


def test_filename_as_language_hint_still_works(sandbox: dict) -> None:
    # A filename outside the roots is fine when real code is supplied
    result = parse_code_to_ast(code=PUBLIC, filename="/somewhere/else/thing.py")
    assert result["language"] == "python"
    assert "public_marker" in result["ast"]["text"]


def test_refusal_does_not_reveal_whether_file_exists(sandbox: dict) -> None:
    real = parse_code_to_ast(filename=str(sandbox["outside"] / "secret.py"))
    fake = parse_code_to_ast(filename=str(sandbox["outside"] / "nope.py"))
    strip = lambda r, n: r["error"].replace(n, "X")  # noqa: E731
    assert strip(real, "secret.py") == strip(fake, "nope.py")


def test_analyze_code_structure_refuses_outside_file(sandbox: dict) -> None:
    result = analyze_code_structure(filename=str(sandbox["outside"] / "secret.py"))
    assert "error" in result
    assert "top_secret_marker" not in str(result)


def test_incremental_parse_refuses_outside_file(sandbox: dict) -> None:
    result = parse_code_to_ast_incremental(
        filename=str(sandbox["outside"] / "secret.py")
    )
    assert "error" in result
    assert "top_secret_marker" not in str(result)


def test_search_pattern_does_not_read_outside_file(sandbox: dict) -> None:
    matches = AstGrepTransformer().search_pattern(
        code="# Placeholder",
        pattern="def $FUNC($$$ARGS)",
        filename=str(sandbox["outside"] / "secret.py"),
    )
    assert matches == []


def test_analyze_source_file_refuses_outside_file(sandbox: dict) -> None:
    result = server.analyze_source_file(
        project_name="guard_test",
        filename=str(sandbox["outside"] / "secret.py"),
        include_summary=False,
    )
    assert "error" in result
    assert "top_secret_marker" not in str(result)


def test_analyze_source_file_refuses_outside_output_folder(sandbox: dict) -> None:
    target = sandbox["outside"] / "dump"
    result = server.analyze_source_file(
        project_name="guard_test",
        code=PUBLIC,
        include_summary=False,
        output_folder=str(target),
    )
    assert "error" in result
    assert not target.exists()


def test_analyze_source_file_writes_inside_root(sandbox: dict) -> None:
    target = sandbox["allowed"] / "reports"
    result = server.analyze_source_file(
        project_name="guard_test",
        filename="ok.py",
        include_summary=False,
        output_folder=str(target),
    )
    assert result.get("status") == "success", result
    assert target.is_dir()


def test_analyze_project_refuses_outside_root(
    sandbox: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_run(name: str) -> Path:
        raise AssertionError("output folder created for a refused project")

    monkeypatch.setattr(
        server.get_output_manager(), "create_analysis_folder", must_not_run
    )
    result = server.analyze_project(
        project_path=str(sandbox["outside"]),
        project_name="guard_test",
        sync_to_db=False,
        include_summary=False,
    )
    assert "error" in result
    assert "processed_files" not in result


def test_analyze_project_skips_symlink_to_outside(
    sandbox: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Keep report output inside the temp folder instead of the real repo
    out = tmp_path / "reports"
    out.mkdir()
    monkeypatch.setattr(
        server.get_output_manager(), "create_analysis_folder", lambda name: out
    )
    monkeypatch.setattr(server, "ANALYZED_PROJECTS_DIR", out)
    (sandbox["allowed"] / "leak.py").symlink_to(sandbox["outside"] / "secret.py")

    result = server.analyze_project(
        project_path=str(sandbox["allowed"]),
        project_name="guard_test",
        sync_to_db=False,
        include_summary=False,
    )
    assert result["processed_files"] == 1  # ok.py only
    assert [Path(f["file"]).name for f in result["failures"]] == ["leak.py"]
    written = "".join(p.read_text() for p in out.rglob("*") if p.is_file())
    assert "top_secret_marker" not in written
    assert "public_marker" in written


# --- cache keys used by the resource URIs -----------------------------------


def test_cache_key_passes_real_hashes() -> None:
    assert safe_cache_key("d41d8cd98f00b204e9800998ecf8427e") == (
        "d41d8cd98f00b204e9800998ecf8427e"
    )


@pytest.mark.parametrize("bad", ["../../etc/passwd", "a/b", "..", "", "a\x00b"])
def test_cache_path_stays_in_cache_dir(bad: str) -> None:
    path = Path(get_cache_path(bad, "ast")).resolve()
    assert path.parent == Path(CACHE_DIR).resolve()

"""
Path guard: keeps every file read and write inside an allow-list of folders.

Why this exists
---------------
Several tools accept a ``filename`` / ``project_path`` / ``output_folder``
argument that comes straight from the calling agent. Without a check, a
confused or prompt-injected agent could ask the server to read any file the
user can read (CWE-22, GitHub issue #2) -- and ``analyze_source_file`` also
forwards file contents to the configured LLM provider.

Policy
------
* Reads are allowed only inside the folders listed in the
  ``AST_MCP_ALLOWED_ROOTS`` environment variable (separated by ``:`` on
  Linux/macOS, ``;`` on Windows). If the variable is unset, only the current
  working directory is allowed.
* Paths are fully resolved first (``~``, ``..`` and symlinks), so a link that
  points outside an allowed folder is rejected.
* Files that usually hold secrets (``.env``, private keys, ``.ssh`` ...) are
  refused even inside an allowed folder.
* Writes are allowed inside the same folders plus the server's own
  ``analyzed_projects`` output folder.
"""

import fnmatch
import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

# Load .env from project root so AST_MCP_ALLOWED_ROOTS can be set there.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed, rely on system env vars

ALLOWED_ROOTS_ENV = "AST_MCP_ALLOWED_ROOTS"

# Directory names that are refused anywhere in a path.
SENSITIVE_DIR_NAMES = frozenset({".ssh", ".gnupg", ".aws", ".kube"})

# File name patterns that are refused even inside an allowed root.
SENSITIVE_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".netrc",
    ".pgpass",
    ".npmrc",
    ".pypirc",
)

_CACHE_KEY_RE = re.compile(r"[A-Za-z0-9]{1,128}")


class PathNotAllowedError(PermissionError):
    """Raised when a requested path is outside the allow-list or is sensitive.

    Subclasses PermissionError (an OSError) so existing ``except OSError``
    handlers around file reads turn it into a normal error response.
    """


def get_allowed_roots() -> List[Path]:
    """Return the resolved folders that tools are allowed to read from.

    Read from the environment on every call so a changed setting (or a test)
    takes effect without restarting.
    """
    raw = os.environ.get(ALLOWED_ROOTS_ENV, "")
    entries = [e.strip() for e in raw.split(os.pathsep) if e.strip()]
    if not entries:
        entries = [os.getcwd()]
    roots: List[Path] = []
    for entry in entries:
        try:
            roots.append(Path(entry).expanduser().resolve())
        except (OSError, ValueError, RuntimeError):
            continue  # ignore an unusable entry rather than widen access
    return roots


def _is_within(path: Path, roots: Iterable[Path]) -> bool:
    """True if ``path`` is one of ``roots`` or lies underneath one."""
    return any(path == root or root in path.parents for root in roots)


def _is_sensitive(path: Path) -> bool:
    """True if the path looks like a credentials file or folder."""
    if any(part in SENSITIVE_DIR_NAMES for part in path.parts):
        return True
    return any(fnmatch.fnmatch(path.name, pat) for pat in SENSITIVE_FILE_PATTERNS)


def resolve_allowed_path(
    path: str, extra_roots: Optional[Iterable[Path]] = None
) -> Path:
    """Resolve ``path`` and confirm it is inside the allow-list.

    Args:
        path: Path supplied by the caller (relative, absolute or ``~``).
        extra_roots: Additional folders to accept for this one check.

    Returns:
        The fully resolved path (symlinks and ``..`` removed).

    Raises:
        PathNotAllowedError: if the path is empty, malformed, outside every
            allowed folder, or matches a sensitive name.
    """
    if not path or not str(path).strip():
        raise PathNotAllowedError("Empty path")
    try:
        resolved = Path(str(path)).expanduser().resolve()
    except (OSError, ValueError, RuntimeError) as e:
        raise PathNotAllowedError(f"Invalid path: {e}") from e

    roots = get_allowed_roots()
    if extra_roots:
        roots = roots + [Path(r).resolve() for r in extra_roots]

    if not _is_within(resolved, roots):
        raise PathNotAllowedError(
            f"Path is outside the allowed folders. Add its folder to "
            f"{ALLOWED_ROOTS_ENV} to permit it."
        )
    if _is_sensitive(resolved):
        raise PathNotAllowedError(
            "Path looks like a credentials file and is never read by this server."
        )
    return resolved


def read_source_file(path: str) -> str:
    """Read a text file after confirming it is inside the allow-list.

    Raises:
        PathNotAllowedError: if the path is not permitted.
        OSError: if the file cannot be read.
    """
    return resolve_allowed_path(path).read_text(encoding="utf-8")


def safe_cache_key(key: str) -> str:
    """Return a cache key that is safe to embed in a file name.

    Real keys are hex digests and pass through unchanged. Anything else
    (slashes, dots ...) is replaced by its own digest, so it can never point
    outside the cache folder and simply results in a cache miss.
    """
    if _CACHE_KEY_RE.fullmatch(key or ""):
        return key
    return hashlib.md5(
        (key or "").encode("utf-8", "replace"), usedforsecurity=False
    ).hexdigest()

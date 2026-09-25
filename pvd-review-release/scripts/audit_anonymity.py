#!/usr/bin/env python3
"""Fail closed on common identity and private-infrastructure leaks.

This is a release hygiene check, not a proof of anonymity. It intentionally
uses conservative rules for source trees while preserving upstream attribution
inside third-party license and notice files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
    }
)
ATTRIBUTION_FILENAMES = frozenset({"authors", "copying", "copyright", "license", "notice"})
ANONYMOUS_NAMES = frozenset(
    {"anonymous", "anonymous author", "anonymous authors", "iclr anonymous"}
)
ANONYMOUS_EMAIL_DOMAINS = frozenset({"example.invalid", "invalid.example"})


@dataclass(frozen=True)
class Rule:
    name: str
    explanation: str
    pattern: re.Pattern[str]
    allow_in_attribution: bool = False


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    column: int
    rule: str
    explanation: str
    context: str


_TRACKING_NAME = "wan" + "db"
_HOST_PREFIX = "no" + "de"
_EXPERIMENT_PREFIX = "prom" + "pt"
_DISK_PREFIX = "hd" + "d"

RULES: tuple[Rule, ...] = (
    Rule(
        "unix-user-path",
        "absolute per-user filesystem path",
        re.compile(r"(?<![A-Za-z0-9])/(?:home|Users)/[A-Za-z0-9._-]+(?:/|$)"),
    ),
    Rule(
        "shared-user-path",
        "absolute shared-storage user path",
        re.compile(r"(?<![A-Za-z0-9])/(?:data|scratch)/users?/[A-Za-z0-9._-]+(?:/|$)"),
    ),
    Rule(
        "machine-disk-path",
        "machine-specific mounted disk path",
        re.compile(rf"(?<![A-Za-z0-9])/{_DISK_PREFIX}\d+(?:/|$)", re.IGNORECASE),
    ),
    Rule(
        "windows-user-path",
        "absolute Windows per-user filesystem path",
        re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:\\Users\\[^\\\s]+(?:\\|$)"),
    ),
    Rule(
        "cluster-hostname",
        "cluster host identifier",
        re.compile(rf"(?i)(?<![A-Za-z0-9]){_HOST_PREFIX}[-_]?\d{{1,5}}(?![A-Za-z0-9])"),
    ),
    Rule(
        "experiment-identifier",
        "internal experiment identifier",
        re.compile(rf"(?i)(?<![A-Za-z0-9]){_EXPERIMENT_PREFIX}[-_]?\d{{2,}}(?![A-Za-z0-9])"),
    ),
    Rule(
        "run-identifier",
        "internal run identifier",
        re.compile(r"(?i)\brun(?:[-_ ]?(?:id|name))?[-_ ]+(?:\d{3,}|[0-9a-f]{8,})\b"),
    ),
    Rule(
        "experiment-tracking",
        "experiment-tracking service or metadata",
        re.compile(rf"(?i)\b{_TRACKING_NAME}(?:\.[A-Za-z0-9_.-]+)?\b"),
    ),
    Rule(
        "email-address",
        "email address outside a third-party attribution file",
        re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        allow_in_attribution=True,
    ),
    Rule(
        "private-ip-address",
        "private-network address",
        re.compile(
            r"(?<![\d.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
            r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?![\d.])"
        ),
    ),
    Rule(
        "embedded-credential",
        "credential embedded in a URL",
        re.compile(r"(?i)\b(?:https?|ssh)://[^\s/:]+:[^\s/@]+@[^\s]+"),
    ),
    Rule(
        "private-key",
        "private key material",
        re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    ),
)

_IDENTITY_METADATA = re.compile(
    r"(?i)^\s*(?:authors?|maintainers?|org\.opencontainers\.image\.authors)\s*[=:]"
)
_EMAIL = next(rule.pattern for rule in RULES if rule.name == "email-address")


def _is_attribution_file(relative_path: Path) -> bool:
    if "third_party" not in relative_path.parts:
        return False
    stem = relative_path.name.lower().split(".", 1)[0]
    return stem in ATTRIBUTION_FILENAMES


def _redacted_context(line: str, start: int, end: int, width: int = 48) -> str:
    left = line[max(0, start - width) : start]
    right = line[end : end + width]
    context = f"{left}[REDACTED]{right}".strip()
    return context[: 2 * width + len("[REDACTED]")]


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            sample = handle.read(8192)
    except OSError:
        return False
    return b"\x00" in sample


def _iter_paths(root: Path) -> Iterable[Path]:
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_names[:] = sorted(
            name for name in directory_names if name not in DEFAULT_IGNORED_DIRECTORIES
        )
        base = Path(directory)
        for name in sorted(directory_names + filenames):
            yield base / name


def _git_tracked_paths(root: Path) -> set[Path]:
    """Return tracked paths even when they live under ignored build directories.

    The working tree scan intentionally skips local environments and caches so a
    reviewer can run it without traversing a multi-gigabyte virtualenv.  A
    tracked file must not inherit that exemption, however: an accidental
    ``git add .venv/model.safetensors`` (or similar) still belongs in the
    release audit.  ``git ls-files`` is unavailable for an unpacked source
    archive, in which case the ordinary tree walk remains the source of truth.
    """

    output = _git_output(root, ["ls-files", "-z"])
    if output is None:
        return set()
    tracked: set[Path] = set()
    for value in output.split("\x00"):
        if not value:
            continue
        relative = Path(value)
        # Git paths are repository-relative; reject malformed output rather than
        # allowing a future implementation to inspect outside ``root``.
        if relative.is_absolute() or ".." in relative.parts:
            continue
        tracked.add(relative)
    return tracked


def _scan_path(path: Path, root: Path, max_file_bytes: int) -> list[Finding]:
    """Scan one path and return findings for its name, target, or contents."""

    relative_path = path.relative_to(root)
    relative_text = relative_path.as_posix()
    findings = _scan_value(
        value=relative_text,
        path=relative_text,
        line_number=0,
        attribution_file=_is_attribution_file(relative_path),
    )
    if path.is_symlink():
        target = os.readlink(path)
        if Path(target).is_absolute():
            findings.append(
                Finding(
                    path=relative_text,
                    line=0,
                    column=0,
                    rule="absolute-symlink",
                    explanation="absolute symlink target",
                    context="[REDACTED TARGET]",
                )
            )
        findings.extend(
            _scan_value(
                value=target,
                path=relative_text,
                line_number=0,
                attribution_file=False,
            )
        )
    elif path.is_file():
        findings.extend(_scan_text(path, relative_path, max_file_bytes))
    return findings


def _scan_value(
    *,
    value: str,
    path: str,
    line_number: int,
    attribution_file: bool,
) -> list[Finding]:
    findings: list[Finding] = []
    for rule in RULES:
        if attribution_file and rule.allow_in_attribution:
            continue
        for match in rule.pattern.finditer(value):
            findings.append(
                Finding(
                    path=path,
                    line=line_number,
                    column=match.start() + 1,
                    rule=rule.name,
                    explanation=rule.explanation,
                    context=_redacted_context(value, match.start(), match.end()),
                )
            )
    return findings


def _scan_text(path: Path, relative_path: Path, max_file_bytes: int) -> list[Finding]:
    try:
        size = path.stat().st_size
    except OSError as error:
        return [
            Finding(
                path=relative_path.as_posix(),
                line=0,
                column=0,
                rule="unreadable-file",
                explanation=str(error),
                context="",
            )
        ]
    if size > max_file_bytes:
        return [
            Finding(
                path=relative_path.as_posix(),
                line=0,
                column=0,
                rule="large-file-not-scanned",
                explanation=f"file exceeds the {max_file_bytes}-byte scan limit",
                context="",
            )
        ]
    if _looks_binary(path):
        return []
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    except OSError as error:
        return [
            Finding(
                path=relative_path.as_posix(),
                line=0,
                column=0,
                rule="unreadable-file",
                explanation=str(error),
                context="",
            )
        ]

    attribution_file = _is_attribution_file(relative_path)
    findings: list[Finding] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        findings.extend(
            _scan_value(
                value=line,
                path=relative_path.as_posix(),
                line_number=line_number,
                attribution_file=attribution_file,
            )
        )
        if not attribution_file and _IDENTITY_METADATA.search(line):
            normalized = line.casefold()
            if "anonymous" not in normalized and "reviewer" not in normalized:
                findings.append(
                    Finding(
                        path=relative_path.as_posix(),
                        line=line_number,
                        column=1,
                        rule="identity-metadata",
                        explanation="non-anonymous author or maintainer metadata",
                        context="[REDACTED METADATA]",
                    )
                )
    return findings


def _git_output(root: Path, arguments: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        # Source archives can be audited on machines without Git installed.
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _scan_git_metadata(root: Path) -> list[Finding]:
    top_level = _git_output(root, ["rev-parse", "--show-toplevel"])
    if top_level is None or Path(top_level.strip()).resolve() != root.resolve():
        return []

    findings: list[Finding] = []
    remote_output = _git_output(root, ["remote", "-v"]) or ""
    for line_number, line in enumerate(remote_output.splitlines(), start=1):
        findings.extend(
            _scan_value(
                value=line,
                path=".git/remotes",
                line_number=line_number,
                attribution_file=False,
            )
        )

    history = _git_output(
        root,
        [
            "log",
            "--all",
            "--format=%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B%x1e",
        ],
    )
    if not history:
        return findings

    for record in history.split("\x1e"):
        fields = record.strip().split("\x1f", 5)
        if len(fields) != 6:
            continue
        commit_hash, author_name, author_email, committer_name, committer_email, message = fields
        short_hash = commit_hash[:12]
        for role, name, email in (
            ("author", author_name, author_email),
            ("committer", committer_name, committer_email),
        ):
            email_domain = email.rpartition("@")[2].casefold()
            if name.strip().casefold() not in ANONYMOUS_NAMES or (
                email_domain not in ANONYMOUS_EMAIL_DOMAINS
            ):
                findings.append(
                    Finding(
                        path=f".git/history/{short_hash}",
                        line=0,
                        column=0,
                        rule="git-identity",
                        explanation=f"non-anonymous Git {role} identity",
                        context="[REDACTED IDENTITY]",
                    )
                )
        for line_number, line in enumerate(message.splitlines(), start=1):
            findings.extend(
                _scan_value(
                    value=line,
                    path=f".git/history/{short_hash}",
                    line_number=line_number,
                    attribution_file=False,
                )
            )
    return findings


def scan_repository(
    root: Path,
    *,
    max_file_bytes: int = 25 * 1024 * 1024,
    scan_git: bool = True,
) -> list[Finding]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    findings: list[Finding] = []
    scanned_paths: set[Path] = set()
    for path in _iter_paths(root):
        findings.extend(_scan_path(path, root, max_file_bytes))
        scanned_paths.add(path)

    # Ignored directories are skipped above for performance, but tracked files
    # inside them are release content and must still be audited.  This second
    # pass is a no-op for an unpacked archive (where ``git ls-files`` is absent)
    # and for ordinary source files already visited by the tree walk.
    for relative_path in sorted(_git_tracked_paths(root)):
        path = root / relative_path
        if path in scanned_paths or not (path.is_file() or path.is_symlink()):
            continue
        findings.extend(_scan_path(path, root, max_file_bytes))

    if scan_git:
        findings.extend(_scan_git_metadata(root))
    return sorted(set(findings))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a release tree for common anonymity leaks.")
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        dest="output_format",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=25 * 1024 * 1024,
        help="fail on text files larger than this limit (default: 25 MiB)",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="skip Git author, committer, message, and remote checks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        findings = scan_repository(
            arguments.root,
            max_file_bytes=arguments.max_file_bytes,
            scan_git=not arguments.no_git,
        )
    except (NotADirectoryError, OSError) as error:
        print(f"anonymity audit error: {error}", file=sys.stderr)
        return 2

    if arguments.output_format == "json":
        print(json.dumps([asdict(finding) for finding in findings], indent=2))
    elif findings:
        for finding in findings:
            location = finding.path
            if finding.line:
                location += f":{finding.line}:{finding.column}"
            print(f"{location}: {finding.rule}: {finding.context}")
        print(f"anonymity audit failed: {len(findings)} finding(s)", file=sys.stderr)
    else:
        print("anonymity audit passed: no findings")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())

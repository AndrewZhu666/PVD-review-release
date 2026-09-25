#!/usr/bin/env python3
"""Build a deterministic, anonymous source archive from the release tree.

The archive is intentionally made from Git's tracked and non-ignored file set.
Local environments, generated reports, model files, and checkpoints are
rejected rather than silently copied.  The output is suitable for a review
upload, but the hosting account and the final upload URL still need a manual
anonymity check.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

DEFAULT_MAX_MEMBER_BYTES = 8 * 1024 * 1024
BLOCKED_COMPONENTS = frozenset(
    {
        ".git",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "artifacts",
        "build",
        "dist",
        "outputs",
        "runs",
    }
)
BLOCKED_SUFFIXES = frozenset(
    {
        ".7z",
        ".arrow",
        # Binary, media, and simulator assets are never source-review members,
        # even when a small file would fit under the size limit.
        ".avi",
        ".bin",
        ".bmp",
        ".bz2",
        ".cab",
        ".ckpt",
        ".dae",
        ".db",
        ".dylib",
        ".exe",
        ".fbx",
        ".gif",
        ".glb",
        ".gltf",
        ".gz",
        ".h5",
        ".hdf5",
        ".ico",
        ".jpeg",
        ".jpg",
        ".lz4",
        ".mkv",
        ".mjcf",
        ".mtl",
        ".mov",
        ".mp3",
        ".mp4",
        ".npy",
        ".npz",
        ".onnx",
        ".obj",
        ".parquet",
        ".pkl",
        ".pickle",
        ".png",
        ".pyd",
        ".pt",
        ".pth",
        ".ply",
        ".pyc",
        ".pyo",
        ".rar",
        ".safetensors",
        ".sdf",
        ".so",
        ".stl",
        ".tar",
        ".tar.gz",
        ".tgz",
        ".tif",
        ".tiff",
        ".usda",
        ".usdc",
        ".usd",
        ".usdz",
        ".urdf",
        ".wav",
        ".webm",
        ".webp",
        ".xz",
        ".zip",
        ".zst",
        # Credentials and key material must never be copied into a review
        # archive, even when a caller accidentally stages or un-ignores them.
        ".key",
        ".pem",
        ".p12",
        ".pfx",
    }
)
BLOCKED_BASENAMES = frozenset(
    {
        ".env",
        "authorized_keys",
        "credentials",
        "credentials.json",
        "credentials.yaml",
        "credentials.yml",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
    }
)


class ArchiveError(ValueError):
    """Raised when the release tree is not safe to archive."""


def _git_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "--full-name"],
        check=True,
        capture_output=True,
        text=True,
    )
    # ``git ls-files -c`` also reports tracked files that have been deleted
    # from the worktree but are not staged yet.  An archive represents the
    # current worktree, so omit those paths while retaining symlinks (including
    # broken ones) for the validator to reject explicitly.
    paths = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        candidate = root / Path(*PurePosixPath(line).parts)
        if candidate.exists() or candidate.is_symlink():
            paths.append(line)
    if paths != sorted(paths):
        paths.sort()
    if len(paths) != len(set(paths)):
        raise ArchiveError("Git release file list contains duplicate paths")
    return paths


def _run_anonymity_audit(root: Path) -> None:
    """Reject a source tree with known identity or host-path findings.

    The archive builder remains usable as a standalone helper when the audit
    script is absent, but a normal release tree includes it.  Running the
    audit here prevents a clean archive from being produced when a later
    untracked file introduces a double-blind leak.
    """

    audit_script = root / "scripts" / "audit_anonymity.py"
    if not audit_script.is_file():
        return
    result = subprocess.run(
        [sys.executable, str(audit_script), str(root), "--format", "json", "--no-git"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ArchiveError(
            "anonymity audit failed; run scripts/audit_anonymity.py manually before archiving"
        )


def _validate_member(root: Path, relative: str, *, max_member_bytes: int) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ArchiveError(f"unsafe archive path: {relative!r}")
    if BLOCKED_COMPONENTS.intersection(path.parts):
        raise ArchiveError(f"generated/private path is in release set: {relative}")
    basename = path.name.casefold()
    if basename in BLOCKED_BASENAMES or (
        basename.startswith(".env.") and basename != ".env.example"
    ):
        raise ArchiveError(f"credential/private file is in release set: {relative}")
    lowered = relative.casefold()
    if any(lowered.endswith(suffix) for suffix in BLOCKED_SUFFIXES):
        raise ArchiveError(f"binary/archive asset is in release set: {relative}")
    candidate = root / Path(*path.parts)
    if candidate.is_symlink():
        raise ArchiveError(f"symbolic links are not allowed in release archive: {relative}")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ArchiveError(f"archive path escapes repository: {relative}")
    if not resolved.is_file():
        raise ArchiveError(f"release path is not a regular file: {relative}")
    size = resolved.stat().st_size
    if size > max_member_bytes:
        raise ArchiveError(
            f"release member is too large ({size} > {max_member_bytes} bytes): {relative}"
        )
    return resolved


def _archive_info(relative: str, source: Path) -> tarfile.TarInfo:
    info = tarfile.TarInfo(relative)
    info.size = source.stat().st_size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o755 if os.access(source, os.X_OK) else 0o644
    return info


def build_archive(
    root: Path,
    output: Path,
    *,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
) -> list[str]:
    """Build ``output`` and return its normalized member list."""

    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    if not root.is_dir():
        raise ArchiveError(f"repository root is not a directory: {root}")
    if max_member_bytes <= 0:
        raise ArchiveError("max-member-bytes must be positive")
    if output == root or output.is_relative_to(root):
        raise ArchiveError("archive output must be outside the repository root")
    _run_anonymity_audit(root)
    members = _git_files(root)
    sources = {
        relative: _validate_member(root, relative, max_member_bytes=max_member_bytes)
        for relative in members
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for relative in members:
                source = sources[relative]
                with source.open("rb") as handle:
                    archive.addfile(_archive_info(relative, source), handle)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return members


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-member-bytes",
        type=int,
        default=DEFAULT_MAX_MEMBER_BYTES,
        help=f"reject larger files (default: {DEFAULT_MAX_MEMBER_BYTES})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        members = build_archive(
            args.root,
            args.output,
            max_member_bytes=args.max_member_bytes,
        )
    except (ArchiveError, OSError, subprocess.CalledProcessError) as error:
        print(f"release archive: {error}")
        return 2
    print(f"created {args.output} ({len(members)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

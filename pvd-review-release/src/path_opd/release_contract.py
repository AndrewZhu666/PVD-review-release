"""Fail-closed release contracts for assets, panels, and checkpoints.

The benchmark workers intentionally accept caller-owned files.  This module is
the small, dependency-free boundary that keeps those files tied to the public
release contract without recording host paths in reports.  It is safe to use
from CPU-only dry-runs and from the benchmark workers after their heavy imports.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from path_opd.config import DEFAULT_CONFIG_DIRECTORY, ConfigError, load_json_object

CONTRACT_SCHEMA = "path-opd-release-contract-v1"
CHECKPOINT_SCHEMA = "path-opd-checkpoint-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PANEL_MODES = frozenset({"formal", "custom"})

# The benchmark workers depend on the small host patch shipped under
# ``integrations/rlinf``.  Keep this gate dependency-light so it can run before
# importing CUDA, OpenPI, or a simulator.
PINNED_RLINF_REVISION = "fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c"
RLINF_HOST_PATCH_SHA256 = "30296071647c0fcdd36d7b12c08fd5d627b5c2f03e8ac73360def51784b30d6f"
RLINF_OPENPI_VERSION = "0.1.1"
_RLINF_COMMON_HOST_FILES = (
    "rlinf/models/embodiment/action_contract.py",
    "rlinf/models/embodiment/openpi/openpi_action_model.py",
)
_RLINF_CALVIN_HOST_FILES = (
    "rlinf/envs/calvin/calvin_gym_env.py",
    "rlinf/envs/calvin/venv.py",
)


class ReleaseContractError(ValueError):
    """Raised when a release artifact is not tied to the declared contract."""


def _exact_patch_is_applied(checkout: Path, patch: Path) -> bool:
    """Return whether Git can cleanly reverse the complete release patch."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), "apply", "--reverse", "--check", str(patch)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReleaseContractError(
            "could not verify the applied RLinf host patch with Git"
        ) from error
    return completed.returncode == 0


def validate_rlinf_host_support(
    checkout: Path | str,
    *,
    benchmark: str,
    require_pinned_revision: bool = True,
    patch_path: Path | str | None = None,
    require_openpi_distribution: bool = False,
) -> dict[str, Any]:
    """Validate the caller's RLinf checkout before heavy runtime imports.

    The release patch is intentionally not applied by the benchmark workers:
    mutating a caller-owned checkout would be surprising and would obscure
    provenance.  Instead, this gate checks the pinned Git baseline and the
    patch-created files/symbols that the workers actually consume.  The return
    value is path-free so it can be written into a public run summary.
    """

    if benchmark not in {"maniskill", "calvin_abc_d", "metaworld_mt50"}:
        raise ReleaseContractError(f"unsupported RLinf benchmark {benchmark!r}")
    patch_digest: str | None = None
    patch: Path | None = None
    if patch_path is not None:
        patch = Path(patch_path).expanduser().resolve(strict=False)
        if not patch.is_file():
            raise ReleaseContractError("the release host-support patch is missing")
        patch_digest = sha256_file(patch)
        if patch_digest != RLINF_HOST_PATCH_SHA256:
            raise ReleaseContractError(
                "the release host-support patch SHA-256 does not match the declared digest"
            )
    openpi_version: str | None = None
    if require_openpi_distribution:
        try:
            openpi_version = importlib.metadata.version("rlinf-openpi")
        except importlib.metadata.PackageNotFoundError as error:
            raise ReleaseContractError(
                f"rlinf-openpi=={RLINF_OPENPI_VERSION} is not installed"
            ) from error
        if openpi_version != RLINF_OPENPI_VERSION:
            raise ReleaseContractError(
                "rlinf-openpi version does not match the release contract: "
                f"expected {RLINF_OPENPI_VERSION}, got {openpi_version}"
            )
    root = Path(checkout).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ReleaseContractError("RLinf checkout is not a directory")
    revision = git_revision(root)
    if require_pinned_revision and revision != PINNED_RLINF_REVISION:
        observed = revision or "unavailable"
        raise ReleaseContractError(
            "RLinf checkout revision is not the pinned release baseline: "
            f"expected {PINNED_RLINF_REVISION}, got {observed}"
        )
    exact_patch_application: bool | None = None
    if patch is not None:
        exact_patch_application = _exact_patch_is_applied(root, patch)
        if not exact_patch_application:
            raise ReleaseContractError(
                "RLinf checkout does not contain the complete release host patch"
            )

    required = list(_RLINF_COMMON_HOST_FILES)
    if benchmark == "calvin_abc_d":
        required.extend(_RLINF_CALVIN_HOST_FILES)
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise ReleaseContractError(
            "RLinf checkout is missing the release host patch; expected: "
            + ", ".join(missing)
        )

    openpi_source = (root / _RLINF_COMMON_HOST_FILES[1]).read_text(
        encoding="utf-8", errors="replace"
    )
    action_contract_source = (root / _RLINF_COMMON_HOST_FILES[0]).read_text(
        encoding="utf-8", errors="replace"
    )
    markers = {
        "valid_action_contract": "class ValidActionContract" in action_contract_source,
        "cropped_openpi_actions": "valid_action_contract" in openpi_source,
        "deterministic_initial_noise": "initial_noise" in openpi_source,
    }
    if benchmark == "calvin_abc_d":
        calvin_source = (root / _RLINF_CALVIN_HOST_FILES[0]).read_text(
            encoding="utf-8", errors="replace"
        )
        venv_source = (root / _RLINF_CALVIN_HOST_FILES[1]).read_text(
            encoding="utf-8", errors="replace"
        )
        markers.update(
            {
                "calvin_checkpoint_state": "capture_checkpoint_state" in calvin_source,
                "calvin_subprocess_serialization": "restore_from_storage" in venv_source,
            }
        )
    missing_markers = sorted(name for name, present in markers.items() if not present)
    if missing_markers:
        raise ReleaseContractError(
            "RLinf checkout does not contain the required release host interfaces: "
            + ", ".join(missing_markers)
        )
    return {
        "pinned_revision": PINNED_RLINF_REVISION,
        "observed_revision": revision,
        "host_patch_sha256": RLINF_HOST_PATCH_SHA256,
        "host_patch_file_sha256": patch_digest,
        "exact_patch_application": exact_patch_application,
        "rlinf_openpi_version": openpi_version,
        "host_patch_interfaces": markers,
        "qualification_gate": "pinned_rlinf_and_host_patch",
    }


@dataclass(frozen=True)
class SourceProvenance:
    """Public source metadata for one dependency or external artifact."""

    name: str
    url: str | None
    revision: str | None
    license: str | None
    sha256: str | None
    redistribution: str
    sha256_scope: str = "artifact"

    def as_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "url": self.url,
            "revision": self.revision,
            "license": self.license,
            "sha256": self.sha256,
            "redistribution": self.redistribution,
            "sha256_scope": self.sha256_scope,
        }


@dataclass(frozen=True)
class PanelContract:
    """Expected digest and semantics for a benchmark's formal panel."""

    benchmark: str
    asset_id: str | None
    protocol: str
    rows: int
    expected_sha256: str | None
    algorithm: str = "sha256_file"

    @property
    def has_formal_digest(self) -> bool:
        return self.expected_sha256 is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "asset_id": self.asset_id,
            "protocol": self.protocol,
            "rows": self.rows,
            "expected_sha256": self.expected_sha256,
            "algorithm": self.algorithm,
            "formal_digest_available": self.has_formal_digest,
        }


@dataclass(frozen=True)
class PanelVerification:
    """Result of binding one panel to either the formal or custom contract."""

    benchmark: str
    mode: str
    formal: bool
    actual_sha256: str
    expected_sha256: str | None
    rows: int
    protocol: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "mode": self.mode,
            "formal": self.formal,
            "custom": not self.formal,
            "actual_sha256": self.actual_sha256,
            "expected_sha256": self.expected_sha256,
            "rows": self.rows,
            "protocol": self.protocol,
        }


def _require_sha256(value: str, label: str) -> str:
    normalized = value.lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ReleaseContractError(f"{label} must be 64 lowercase hexadecimal digits")
    return normalized


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON for stable release digests (including a final newline)."""

    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path | str) -> str:
    artifact = Path(path)
    if not artifact.is_file():
        raise ReleaseContractError(f"expected a regular file for hashing: {artifact}")
    digest = hashlib.sha256()
    try:
        with artifact.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ReleaseContractError(f"could not hash {artifact}: {error}") from error
    return digest.hexdigest()


def sha256_canonical_json(path: Path | str) -> str:
    artifact = Path(path)
    try:
        payload = load_json_object(artifact)
    except (OSError, ConfigError) as error:
        raise ReleaseContractError(f"could not parse JSON asset {artifact}: {error}") from error
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _manifest_path(config_directory: Path | str, benchmark: str) -> Path:
    return Path(config_directory) / "assets" / f"{benchmark}.json"


def formal_panel_contract(
    benchmark: str,
    *,
    config_directory: Path | str = DEFAULT_CONFIG_DIRECTORY,
) -> PanelContract:
    """Read the formal panel digest from the benchmark asset manifest.

    The manifest is the single source of truth for the two file-backed formal
    panels.  MetaWorld's panel is generated from task metadata at runtime and
    deliberately has no file digest in this release; it must therefore be
    labelled custom until a public panel artifact is published.
    """

    try:
        config = load_json_object(Path(config_directory) / f"{benchmark}.json")
        assets = load_json_object(_manifest_path(config_directory, benchmark))["assets"]
    except (OSError, ConfigError, KeyError, TypeError) as error:
        raise ReleaseContractError(
            f"could not load panel contract for {benchmark!r}: {error}"
        ) from error
    if not isinstance(config, dict) or not isinstance(assets, list):
        raise ReleaseContractError(f"invalid configuration for benchmark {benchmark!r}")
    panel_id = config.get("evaluation", {}).get("panel_asset_id")
    rows = config.get("evaluation", {}).get("rows_per_model")
    protocol = config.get("evaluation", {}).get("protocol")
    if not isinstance(rows, int) or not isinstance(protocol, str):
        raise ReleaseContractError(f"benchmark {benchmark!r} has invalid evaluation contract")
    if panel_id is None:
        return PanelContract(benchmark, None, protocol, rows, None)
    for raw_asset in assets:
        if isinstance(raw_asset, dict) and raw_asset.get("id") == panel_id:
            integrity = raw_asset.get("integrity")
            if not isinstance(integrity, dict):
                break
            digest = integrity.get("sha256")
            algorithm = integrity.get("algorithm", "sha256_file")
            if not isinstance(algorithm, str):
                raise ReleaseContractError(f"panel {benchmark!r} has invalid digest algorithm")
            if digest is not None:
                digest = _require_sha256(str(digest), f"{benchmark} formal panel digest")
            return PanelContract(benchmark, panel_id, protocol, rows, digest, algorithm)
    raise ReleaseContractError(
        f"benchmark {benchmark!r} references missing panel asset {panel_id!r}"
    )


def verify_panel_digest(
    benchmark: str,
    actual_sha256: str,
    *,
    mode: str = "formal",
    supplied_sha256: str | None = None,
    rows: int | None = None,
    config_directory: Path | str = DEFAULT_CONFIG_DIRECTORY,
) -> PanelVerification:
    """Bind an observed panel digest to the formal contract or explicit custom mode.

    Formal mode is fail-closed: a benchmark must have a known public digest and
    the observed digest must equal it.  Custom mode requires an explicit caller
    opt-in and is never marked as a paper-panel result.
    """

    if mode not in _PANEL_MODES:
        raise ReleaseContractError(f"panel mode must be one of {sorted(_PANEL_MODES)!r}")
    contract = formal_panel_contract(benchmark, config_directory=config_directory)
    actual = _require_sha256(actual_sha256, "observed panel SHA-256")
    if supplied_sha256 is not None:
        supplied = _require_sha256(supplied_sha256, "supplied panel SHA-256")
        if supplied != actual:
            raise ReleaseContractError(
                f"supplied panel SHA-256 {supplied} does not match observed {actual}"
            )
    if rows is not None and rows != contract.rows:
        raise ReleaseContractError(
            f"panel row count {rows} does not match {benchmark} contract ({contract.rows})"
        )
    if mode == "formal":
        if not contract.has_formal_digest:
            raise ReleaseContractError(
                f"{benchmark} has no published formal panel digest; pass --panel-mode custom "
                "and report the result as non-formal"
            )
        if actual != contract.expected_sha256:
            raise ReleaseContractError(
                f"{benchmark} panel digest is not the formal release panel: "
                f"expected {contract.expected_sha256}, got {actual}"
            )
        return PanelVerification(
            benchmark,
            "formal",
            True,
            actual,
            contract.expected_sha256,
            contract.rows,
            contract.protocol,
        )
    if supplied_sha256 is None:
        raise ReleaseContractError(
            "custom panel mode requires --panel-sha256 so the output records its exact bytes"
        )
    return PanelVerification(
        benchmark,
        "custom",
        False,
        actual,
        contract.expected_sha256,
        contract.rows,
        contract.protocol,
    )


def _iter_files(root: Path, *, include: Iterable[str] | None = None) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise ReleaseContractError(f"path does not exist: {root}")
    prefixes = tuple(include or ())
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        ignored = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
        if any(part in ignored for part in candidate.parts):
            continue
        if prefixes and not any(
            relative == prefix or relative.startswith(prefix.rstrip("/") + "/")
            for prefix in prefixes
        ):
            continue
        files.append(candidate)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def tree_sha256(root: Path | str, *, include: Iterable[str] | None = None) -> str:
    """Hash a source tree by relative names and bytes, independent of host paths."""

    path = Path(root)
    files = _iter_files(path, include=include)
    digest = hashlib.sha256()
    base = path if path.is_dir() else path.parent
    for item in files:
        relative = item.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        # Record the byte length without loading a potentially multi-GB model
        # checkpoint into memory before the streaming hash below.
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path | str) -> str | None:
    """Return a checkout revision without leaking its absolute path."""

    candidate = Path(path).expanduser()
    try:
        result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision if re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) else None


def build_checkpoint_manifest(
    *,
    benchmark: str,
    method: str,
    seed: int,
    world_size: int,
    update: int,
    contract: Mapping[str, Any],
    assets: Mapping[str, str],
    code: Mapping[str, str | None],
    panel: PanelVerification | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a path-free checkpoint manifest with all resume identity fields."""

    if not benchmark or not method or seed < 0 or world_size < 1 or update < 0:
        raise ReleaseContractError("invalid checkpoint identity fields")
    normalized_assets = {
        str(name): _require_sha256(str(value), f"asset {name} SHA-256")
        for name, value in sorted(assets.items())
    }
    normalized_code: dict[str, str | None] = {}
    for name, value in sorted(code.items()):
        if value is not None:
            value = _require_sha256(value, f"code {name} SHA-256") if len(value) == 64 else value
        normalized_code[str(name)] = value
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "contract_schema": CONTRACT_SCHEMA,
        "benchmark": benchmark,
        "method": method,
        "seed": seed,
        "world_size": world_size,
        "optimizer_update": update,
        "contract": json.loads(json.dumps(contract, sort_keys=True, default=str)),
        "asset_sha256": normalized_assets,
        "code": normalized_code,
        "panel": panel.as_dict() if panel is not None else None,
    }
    if extra:
        payload["extra"] = json.loads(json.dumps(dict(extra), sort_keys=True, default=str))
    return payload


def validate_checkpoint_manifest(
    manifest: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    assets: Mapping[str, str] | None = None,
    code: Mapping[str, str | None] | None = None,
    panel: PanelVerification | None = None,
) -> None:
    """Reject a resume checkpoint whose identity differs from the current run."""

    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise ReleaseContractError("checkpoint manifest schema mismatch")
    if manifest.get("contract_schema") != CONTRACT_SCHEMA:
        raise ReleaseContractError("checkpoint release-contract schema mismatch")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ReleaseContractError(
                f"checkpoint {key} mismatch: {manifest.get(key)!r} != {value!r}"
            )
    recorded_assets = manifest.get("asset_sha256")
    if assets is not None:
        if not isinstance(recorded_assets, Mapping):
            raise ReleaseContractError("checkpoint lacks asset SHA-256 manifest")
        for name, value in assets.items():
            actual = _require_sha256(str(value), f"asset {name} SHA-256")
            if recorded_assets.get(name) != actual:
                raise ReleaseContractError(
                    f"checkpoint asset {name} mismatch: {recorded_assets.get(name)!r} != {actual!r}"
                )
    if code is not None:
        recorded_code = manifest.get("code")
        if not isinstance(recorded_code, Mapping):
            raise ReleaseContractError("checkpoint lacks code provenance")
        for name, value in code.items():
            if recorded_code.get(name) != value:
                raise ReleaseContractError(
                    f"checkpoint code {name} mismatch: {recorded_code.get(name)!r} != {value!r}"
                )
    if panel is not None and manifest.get("panel") != panel.as_dict():
        raise ReleaseContractError("checkpoint panel contract mismatch")


def load_provenance_manifest(path: Path | str) -> tuple[SourceProvenance, ...]:
    """Load and validate the public dependency provenance table."""

    try:
        payload = load_json_object(path)
    except (OSError, ConfigError) as error:
        raise ReleaseContractError(f"could not load provenance manifest {path}: {error}") from error
    if payload.get("schema") != CONTRACT_SCHEMA or not isinstance(payload.get("sources"), list):
        raise ReleaseContractError("invalid provenance manifest schema")
    result: list[SourceProvenance] = []
    for index, item in enumerate(payload["sources"]):
        if not isinstance(item, dict):
            raise ReleaseContractError(f"provenance source {index} is not an object")
        required = {"name", "url", "revision", "license", "sha256", "redistribution"}
        optional = {"sha256_scope"}
        if set(item) - required - optional or not required <= set(item):
            raise ReleaseContractError(f"provenance source {index} fields do not match schema")
        for key in ("name", "redistribution"):
            if not isinstance(item[key], str) or not item[key]:
                raise ReleaseContractError(f"provenance source {index}.{key} is invalid")
        for key in ("url", "revision", "license"):
            if item[key] is not None and (not isinstance(item[key], str) or not item[key]):
                raise ReleaseContractError(f"provenance source {index}.{key} is invalid")
        if item["url"] is not None and not str(item["url"]).startswith("https://"):
            raise ReleaseContractError(f"provenance source {index}.url must use HTTPS")
        if item["sha256"] is not None:
            _require_sha256(str(item["sha256"]), f"provenance source {index}.sha256")
        scope = item.get("sha256_scope", "artifact")
        if not isinstance(scope, str) or not scope:
            raise ReleaseContractError(f"provenance source {index}.sha256_scope is invalid")
        normalized = dict(item)
        normalized["sha256_scope"] = scope
        result.append(SourceProvenance(**normalized))
    return tuple(result)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONTRACT_SCHEMA",
    "PINNED_RLINF_REVISION",
    "RLINF_HOST_PATCH_SHA256",
    "PanelContract",
    "PanelVerification",
    "ReleaseContractError",
    "SourceProvenance",
    "build_checkpoint_manifest",
    "canonical_json_bytes",
    "formal_panel_contract",
    "git_revision",
    "load_provenance_manifest",
    "sha256_canonical_json",
    "sha256_file",
    "tree_sha256",
    "validate_checkpoint_manifest",
    "validate_rlinf_host_support",
    "verify_panel_digest",
]

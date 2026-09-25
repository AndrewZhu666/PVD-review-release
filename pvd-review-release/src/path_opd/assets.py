"""Asset manifests, explicit path resolution, and SHA-256 verification."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from path_opd.config import (
    DEFAULT_CONFIG_DIRECTORY,
    PAPER_BENCHMARKS,
    ConfigError,
    load_json_object,
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOCATION_TYPES = frozenset({"repository_relative", "user_parameter"})
_STATUSES = frozenset({"known", "unknown"})
_USES = frozenset({"train", "evaluate", "reproduce_paper_results"})
_METHODS = frozenset({"endpoint_dagger", "path_opd"})


class AssetResolutionError(ValueError):
    """Raised when required asset paths were not supplied or cannot be resolved."""


class AssetIntegrityError(ValueError):
    """Raised when a known asset digest cannot be verified."""


@dataclass(frozen=True)
class AssetLocation:
    type: str
    parameter: str | None
    relative_path: str | None
    public_url_status: str
    public_url: str | None
    # Optional path inside a public repository/archive. ``parameter`` still
    # names the caller-supplied local artifact that is hashed.
    artifact_path: str | None = None
    # Revision is kept beside the public URL because a URL alone is mutable.
    # Defaults preserve compatibility with hand-built test manifests; release
    # manifests carry the fields explicitly.
    revision_status: str = "unknown"
    revision: str | None = None


@dataclass(frozen=True)
class AssetLicense:
    status: str
    identifier: str | None


@dataclass(frozen=True)
class AssetIntegrity:
    status: str
    algorithm: str
    sha256: str | None


@dataclass(frozen=True)
class CheckpointSelector:
    training_seed: int
    method: str


@dataclass(frozen=True)
class Asset:
    id: str
    kind: str
    description: str
    required: bool
    used_by: tuple[str, ...]
    location: AssetLocation
    license: AssetLicense
    integrity: AssetIntegrity
    expected_filename: str | None
    selector: CheckpointSelector | None


@dataclass(frozen=True)
class AssetManifest:
    schema_version: int
    benchmark: str
    assets: tuple[Asset, ...]


def _fail(location: str, message: str) -> None:
    raise ConfigError(f"{location}: {message}")


def _object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail(location, "expected a JSON object")
    if not all(isinstance(key, str) for key in value):
        _fail(location, "object keys must be strings")
    return value


def _strict_fields(
    value: object,
    *,
    required: set[str],
    optional: set[str] | None = None,
    location: str,
) -> dict[str, object]:
    payload = _object(value, location)
    optional = optional or set()
    missing = sorted(required - payload.keys())
    unexpected = sorted(payload.keys() - required - optional)
    if missing:
        _fail(location, f"missing field(s): {', '.join(missing)}")
    if unexpected:
        _fail(location, f"unexpected field(s): {', '.join(unexpected)}")
    return payload


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(location, "expected a non-empty string")
    return value


def _optional_string(value: object, location: str) -> str | None:
    if value is None:
        return None
    return _string(value, location)


def _identifier(value: object, location: str) -> str:
    result = _string(value, location)
    if _IDENTIFIER.fullmatch(result) is None:
        _fail(location, "expected a lower_snake_case identifier")
    return result


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        _fail(location, "expected a boolean")
    return value


def _integer(value: object, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(location, f"expected an integer >= {minimum}")
    return value


def _relative_path(value: object, location: str) -> str:
    result = _string(value, location)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != result or result in {".", ""}:
        _fail(location, "must be a normalized repository-relative path")
    return result


def _parse_location(value: object, location: str) -> AssetLocation:
    payload = _strict_fields(
        value,
        required={
            "type",
            "parameter",
            "relative_path",
            "public_url_status",
            "public_url",
        },
        optional={"artifact_path", "revision_status", "revision"},
        location=location,
    )
    location_type = _string(payload["type"], f"{location}.type")
    if location_type not in _LOCATION_TYPES:
        _fail(
            f"{location}.type",
            f"expected one of {sorted(_LOCATION_TYPES)!r}",
        )
    parameter = _optional_string(payload["parameter"], f"{location}.parameter")
    relative_path = _optional_string(payload["relative_path"], f"{location}.relative_path")
    if location_type == "user_parameter":
        if parameter is None or _IDENTIFIER.fullmatch(parameter) is None:
            _fail(
                f"{location}.parameter",
                "user_parameter locations require a lower_snake_case parameter",
            )
        if relative_path is not None:
            _fail(
                f"{location}.relative_path",
                "must be null for user_parameter locations",
            )
    else:
        if parameter is not None:
            _fail(
                f"{location}.parameter",
                "must be null for repository_relative locations",
            )
        relative_path = _relative_path(
            relative_path,
            f"{location}.relative_path",
        )
    url_status = _string(payload["public_url_status"], f"{location}.public_url_status")
    if url_status not in _STATUSES:
        _fail(f"{location}.public_url_status", "expected 'known' or 'unknown'")
    public_url = _optional_string(payload["public_url"], f"{location}.public_url")
    if url_status == "unknown" and public_url is not None:
        _fail(f"{location}.public_url", "must be null when status is unknown")
    if url_status == "known":
        if public_url is None:
            _fail(f"{location}.public_url", "is required when status is known")
        parsed = urlsplit(public_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
            _fail(
                f"{location}.public_url",
                "must be an HTTPS URL without embedded credentials",
            )
    artifact_path = _optional_string(payload.get("artifact_path"), f"{location}.artifact_path")
    if artifact_path is not None:
        artifact_path = _relative_path(artifact_path, f"{location}.artifact_path")
    revision_status = _string(
        payload.get("revision_status", "unknown"),
        f"{location}.revision_status",
    )
    if revision_status not in _STATUSES:
        _fail(f"{location}.revision_status", "expected 'known' or 'unknown'")
    revision = _optional_string(payload.get("revision"), f"{location}.revision")
    if revision_status == "unknown" and revision is not None:
        _fail(f"{location}.revision", "must be null when status is unknown")
    if revision_status == "known" and revision is None:
        _fail(f"{location}.revision", "is required when status is known")
    return AssetLocation(
        type=location_type,
        parameter=parameter,
        relative_path=relative_path,
        public_url_status=url_status,
        public_url=public_url,
        artifact_path=artifact_path,
        revision_status=revision_status,
        revision=revision,
    )


def _parse_license(value: object, location: str) -> AssetLicense:
    payload = _strict_fields(
        value,
        required={"status", "identifier"},
        location=location,
    )
    status = _string(payload["status"], f"{location}.status")
    if status not in _STATUSES:
        _fail(f"{location}.status", "expected 'known' or 'unknown'")
    identifier = _optional_string(payload["identifier"], f"{location}.identifier")
    if (status == "known") != (identifier is not None):
        _fail(
            f"{location}.identifier",
            "must be set exactly when license status is known",
        )
    return AssetLicense(status=status, identifier=identifier)


def _parse_integrity(value: object, location: str) -> AssetIntegrity:
    payload = _object(value, location)
    required = {"status", "sha256"}
    optional = {"algorithm"}
    missing = sorted(required - payload.keys())
    unexpected = sorted(payload.keys() - required - optional)
    if missing:
        _fail(location, f"missing field(s): {', '.join(missing)}")
    if unexpected:
        _fail(location, f"unexpected field(s): {', '.join(unexpected)}")
    status = _string(payload["status"], f"{location}.status")
    if status not in _STATUSES:
        _fail(f"{location}.status", "expected 'known' or 'unknown'")
    sha256 = _optional_string(payload["sha256"], f"{location}.sha256")
    if status == "known":
        if sha256 is None or _SHA256.fullmatch(sha256) is None:
            _fail(f"{location}.sha256", "expected 64 lowercase hexadecimal digits")
    elif sha256 is not None:
        _fail(f"{location}.sha256", "must be null when status is unknown")
    algorithm = _string(payload.get("algorithm", "sha256_file"), f"{location}.algorithm")
    if algorithm not in {"sha256_file", "sha256_canonical_json"}:
        _fail(
            f"{location}.algorithm",
            "expected 'sha256_file' or 'sha256_canonical_json'",
        )
    return AssetIntegrity(status=status, algorithm=algorithm, sha256=sha256)


def _parse_selector(value: object, location: str) -> CheckpointSelector | None:
    if value is None:
        return None
    payload = _strict_fields(
        value,
        required={"training_seed", "method"},
        location=location,
    )
    method = _string(payload["method"], f"{location}.method")
    if method not in _METHODS:
        _fail(f"{location}.method", f"expected one of {sorted(_METHODS)!r}")
    return CheckpointSelector(
        training_seed=_integer(payload["training_seed"], f"{location}.training_seed"),
        method=method,
    )


def _parse_asset(value: object, index: int) -> Asset:
    location = f"assets[{index}]"
    payload = _strict_fields(
        value,
        required={
            "id",
            "kind",
            "description",
            "required",
            "used_by",
            "location",
            "license",
            "integrity",
            "expected_filename",
            "selector",
        },
        location=location,
    )
    used_by_payload = payload["used_by"]
    if not isinstance(used_by_payload, list) or not used_by_payload:
        _fail(f"{location}.used_by", "expected a non-empty array")
    used_by = tuple(
        _string(item, f"{location}.used_by[{item_index}]")
        for item_index, item in enumerate(used_by_payload)
    )
    if len(used_by) != len(set(used_by)) or not set(used_by) <= _USES:
        _fail(
            f"{location}.used_by",
            f"expected unique values from {sorted(_USES)!r}",
        )
    kind = _identifier(payload["kind"], f"{location}.kind")
    selector = _parse_selector(payload["selector"], f"{location}.selector")
    if (kind == "paper_checkpoint") != (selector is not None):
        _fail(
            f"{location}.selector",
            "is required exactly for paper_checkpoint assets",
        )
    expected_filename = _optional_string(
        payload["expected_filename"], f"{location}.expected_filename"
    )
    if expected_filename is not None and Path(expected_filename).name != expected_filename:
        _fail(f"{location}.expected_filename", "must be a filename, not a path")
    return Asset(
        id=_identifier(payload["id"], f"{location}.id"),
        kind=kind,
        description=_string(payload["description"], f"{location}.description"),
        required=_boolean(payload["required"], f"{location}.required"),
        used_by=used_by,
        location=_parse_location(payload["location"], f"{location}.location"),
        license=_parse_license(payload["license"], f"{location}.license"),
        integrity=_parse_integrity(payload["integrity"], f"{location}.integrity"),
        expected_filename=expected_filename,
        selector=selector,
    )


def validate_asset_payload(value: object) -> AssetManifest:
    """Validate a decoded asset manifest and return immutable records."""

    payload = _strict_fields(
        value,
        required={"schema_version", "benchmark", "assets"},
        location="asset manifest",
    )
    schema_version = _integer(payload["schema_version"], "schema_version", minimum=1)
    if schema_version != 1:
        _fail("schema_version", "only version 1 is supported")
    benchmark = _string(payload["benchmark"], "benchmark")
    if benchmark not in PAPER_BENCHMARKS:
        _fail("benchmark", f"unsupported benchmark {benchmark!r}")
    raw_assets = payload["assets"]
    if not isinstance(raw_assets, list) or not raw_assets:
        _fail("assets", "expected a non-empty array")
    assets = tuple(_parse_asset(item, index) for index, item in enumerate(raw_assets))
    ids = tuple(asset.id for asset in assets)
    if len(ids) != len(set(ids)):
        _fail("assets", "asset ids must be unique")
    parameters = tuple(
        asset.location.parameter for asset in assets if asset.location.parameter is not None
    )
    if len(parameters) != len(set(parameters)):
        _fail("assets", "user parameter names must be unique")
    selectors = tuple(
        (asset.selector.training_seed, asset.selector.method)
        for asset in assets
        if asset.selector is not None
    )
    if len(selectors) != len(set(selectors)):
        _fail("assets", "paper checkpoint selectors must be unique")
    return AssetManifest(
        schema_version=schema_version,
        benchmark=benchmark,
        assets=assets,
    )


def load_asset_manifest(
    benchmark: str,
    *,
    config_directory: Path | str = DEFAULT_CONFIG_DIRECTORY,
) -> AssetManifest:
    """Load one benchmark's asset manifest."""

    if benchmark not in PAPER_BENCHMARKS:
        raise ConfigError(f"unsupported benchmark {benchmark!r}")
    path = Path(config_directory) / "assets" / f"{benchmark}.json"
    manifest = validate_asset_payload(load_json_object(path))
    if manifest.benchmark != benchmark:
        raise ConfigError(f"{path}: benchmark field {manifest.benchmark!r} does not match filename")
    return manifest


def load_all_asset_manifests(
    *,
    config_directory: Path | str = DEFAULT_CONFIG_DIRECTORY,
) -> dict[str, AssetManifest]:
    """Load manifests for exactly the three paper benchmarks."""

    return {
        benchmark: load_asset_manifest(
            benchmark,
            config_directory=config_directory,
        )
        for benchmark in PAPER_BENCHMARKS
    }


def _resolved_asset_path(asset: Asset, candidate: Path) -> Path:
    if candidate.is_dir() and asset.expected_filename is not None:
        candidate = candidate / asset.expected_filename
    return candidate.resolve(strict=False)


def resolve_asset_paths(
    manifest: AssetManifest,
    *,
    supplied: Mapping[str, str | Path],
    repository_root: Path | str | None = None,
    must_exist: bool = True,
) -> dict[str, Path]:
    """Resolve paths without machine-specific defaults or environment fallbacks."""

    root = (
        Path(repository_root).resolve(strict=False)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    resolved: dict[str, Path] = {}
    missing_parameters: list[str] = []
    missing_paths: list[str] = []
    for asset in manifest.assets:
        if asset.location.type == "user_parameter":
            parameter = asset.location.parameter
            assert parameter is not None
            if parameter not in supplied:
                if asset.required:
                    missing_parameters.append(parameter)
                continue
            candidate = Path(supplied[parameter]).expanduser()
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            path = _resolved_asset_path(asset, candidate)
        else:
            relative_path = asset.location.relative_path
            assert relative_path is not None
            path = _resolved_asset_path(asset, root / relative_path)
            try:
                path.relative_to(root)
            except ValueError as error:
                raise AssetResolutionError(
                    f"asset {asset.id!r} escapes the repository root"
                ) from error
        if must_exist and not path.exists():
            missing_paths.append(f"{asset.id}={path}")
            continue
        resolved[asset.id] = path
    if missing_parameters:
        names = ", ".join(sorted(missing_parameters))
        raise AssetResolutionError(f"missing required asset parameters: {names}")
    if missing_paths:
        paths = ", ".join(sorted(missing_paths))
        raise AssetResolutionError(f"asset paths do not exist: {paths}")
    return resolved


def sha256_file(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one regular file without loading it into memory."""

    artifact = Path(path)
    if not artifact.is_file():
        raise AssetIntegrityError(f"asset is not a regular file: {artifact}")
    digest = hashlib.sha256()
    try:
        with artifact.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as error:
        raise AssetIntegrityError(f"could not hash {artifact}: {error}") from error
    return digest.hexdigest()


def sha256_canonical_json_file(path: Path | str) -> str:
    """Hash decoded JSON using sorted keys and compact ASCII serialization."""

    artifact = Path(path)
    payload = load_json_object(artifact)
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def verify_asset_integrity(asset: Asset, path: Path | str) -> bool:
    """Verify an asset with a recorded digest; unknown digests never pass silently."""

    expected = asset.integrity.sha256
    if asset.integrity.status != "known" or expected is None:
        raise AssetIntegrityError(f"asset {asset.id!r} has no known SHA-256 digest")
    if asset.integrity.algorithm == "sha256_file":
        actual = sha256_file(path)
    else:
        actual = sha256_canonical_json_file(path)
    if actual != expected:
        raise AssetIntegrityError(
            f"SHA-256 mismatch for asset {asset.id!r}: expected {expected}, got {actual}"
        )
    return True

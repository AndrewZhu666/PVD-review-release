"""Benchmark contract inspection and fail-closed asset preflight."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from path_opd.assets import (
    Asset,
    AssetManifest,
    load_asset_manifest,
    resolve_asset_paths,
    verify_asset_integrity,
)
from path_opd.config import (
    DEFAULT_CONFIG_DIRECTORY,
    BenchmarkConfig,
    ConfigError,
    load_benchmark_config,
)

PURPOSES = ("train", "evaluate", "reproduce_paper_results")
METHODS = ("endpoint_dagger", "path_opd")
_PARAMETER_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_selector(
    config: BenchmarkConfig,
    *,
    method: str | None,
    seed: int | None,
) -> None:
    if method is not None and method not in config.training.methods:
        raise ConfigError(f"method {method!r} is not configured for benchmark {config.benchmark!r}")
    if seed is not None and seed not in config.training.seeds:
        raise ConfigError(f"seed {seed!r} is not configured for benchmark {config.benchmark!r}")


def select_assets(
    config: BenchmarkConfig,
    manifest: AssetManifest,
    *,
    purpose: str,
    method: str | None = None,
    seed: int | None = None,
) -> AssetManifest:
    """Select only assets needed by one explicit benchmark operation."""

    if manifest.benchmark != config.benchmark:
        raise ConfigError(
            f"asset manifest {manifest.benchmark!r} does not match {config.benchmark!r}"
        )
    if purpose not in PURPOSES:
        raise ConfigError(f"unsupported benchmark purpose {purpose!r}")
    _validate_selector(config, method=method, seed=seed)
    if (method is None) != (seed is None):
        raise ConfigError("method and seed must be supplied together")
    if purpose != "reproduce_paper_results" and method is not None:
        raise ConfigError(
            "method and seed select sealed paper checkpoints only for "
            "purpose 'reproduce_paper_results'"
        )

    selected: list[Asset] = []
    for asset in manifest.assets:
        selector = asset.selector
        if selector is not None:
            if purpose != "reproduce_paper_results":
                continue
            if method is not None and (selector.method != method or selector.training_seed != seed):
                continue
            selected.append(asset)
            continue

        required_uses = {purpose}
        if purpose == "reproduce_paper_results":
            required_uses.add("evaluate")
        if required_uses.intersection(asset.used_by):
            selected.append(asset)

    if not selected:
        raise ConfigError(f"no assets selected for {config.benchmark!r} purpose {purpose!r}")
    return AssetManifest(
        schema_version=manifest.schema_version,
        benchmark=manifest.benchmark,
        assets=tuple(selected),
    )


def describe_benchmark(
    benchmark: str,
    *,
    config_directory: Path | str = DEFAULT_CONFIG_DIRECTORY,
) -> dict[str, Any]:
    """Return the validated paper contract and its explicit asset parameters."""

    config = load_benchmark_config(benchmark, config_directory=config_directory)
    manifest = load_asset_manifest(benchmark, config_directory=config_directory)
    return {
        "benchmark": asdict(config),
        "assets": [
            {
                "id": asset.id,
                "kind": asset.kind,
                "parameter": asset.location.parameter,
                "expected_filename": asset.expected_filename,
                "used_by": list(asset.used_by),
                "selector": asdict(asset.selector) if asset.selector is not None else None,
                "integrity": asdict(asset.integrity),
                "public_url_status": asset.location.public_url_status,
                "public_url": asset.location.public_url,
                "artifact_path": asset.location.artifact_path,
                "revision_status": asset.location.revision_status,
                "revision": asset.location.revision,
                "license_status": asset.license.status,
                "license_identifier": asset.license.identifier,
            }
            for asset in manifest.assets
        ],
    }


def preflight_benchmark(
    benchmark: str,
    *,
    purpose: str,
    supplied: Mapping[str, str | Path],
    method: str | None = None,
    seed: int | None = None,
    config_directory: Path | str = DEFAULT_CONFIG_DIRECTORY,
) -> dict[str, Any]:
    """Resolve required assets and verify every available integrity contract."""

    config = load_benchmark_config(benchmark, config_directory=config_directory)
    manifest = select_assets(
        config,
        load_asset_manifest(benchmark, config_directory=config_directory),
        purpose=purpose,
        method=method,
        seed=seed,
    )
    expected_parameters = {
        asset.location.parameter
        for asset in manifest.assets
        if asset.location.parameter is not None
    }
    unexpected = sorted(set(supplied) - expected_parameters)
    if unexpected:
        raise ConfigError(f"unexpected asset parameter(s): {', '.join(unexpected)}")

    resolved = resolve_asset_paths(manifest, supplied=supplied)
    entries: list[dict[str, Any]] = []
    unverified: list[str] = []
    metadata_gaps: list[dict[str, str]] = []
    for asset in manifest.assets:
        path = resolved.get(asset.id)
        if path is None:
            entries.append(
                {
                    "id": asset.id,
                    "kind": asset.kind,
                    "parameter": asset.location.parameter,
                    "path": None,
                    "integrity": "not_supplied",
                    "integrity_algorithm": asset.integrity.algorithm,
                    "sha256": asset.integrity.sha256,
                }
            )
            continue
        if asset.integrity.status == "known":
            verify_asset_integrity(asset, path)
            integrity = "verified"
        else:
            integrity = "unknown"
            unverified.append(asset.id)
        if (
            asset.location.public_url_status != "known"
            or asset.location.revision_status != "known"
            or asset.license.status != "known"
        ):
            metadata_gaps.append(
                {
                    "asset": asset.id,
                    "public_url": asset.location.public_url_status,
                    "revision": asset.location.revision_status,
                    "license": asset.license.status,
                }
            )
        entries.append(
            {
                "id": asset.id,
                "kind": asset.kind,
                "parameter": asset.location.parameter,
                "path": path.as_posix(),
                "integrity": integrity,
                "integrity_algorithm": asset.integrity.algorithm,
                "sha256": asset.integrity.sha256,
            }
        )

    return {
        "status": "PASS_WITH_UNVERIFIED_ASSETS" if unverified else "PASS",
        "benchmark": benchmark,
        "purpose": purpose,
        "method": method,
        "training_seed": seed,
        "asset_count": len(entries),
        "assets": entries,
        "unverified_assets": unverified,
        "public_metadata_gaps": metadata_gaps,
        "warning": (
            "Existence without a recorded digest is not byte-level provenance."
            if unverified
            else None
        ),
    }


def parse_asset_assignments(values: list[str]) -> dict[str, Path]:
    """Parse repeated lower_snake_case=PATH command-line assignments."""

    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ConfigError(f"asset assignment must be NAME=PATH, got {value!r}")
        if _PARAMETER_NAME.fullmatch(name) is None:
            raise ConfigError(f"asset parameter must be lower_snake_case, got {name!r}")
        if name in result:
            raise ConfigError(f"duplicate asset assignment for {name!r}")
        result[name] = Path(path)
    return result

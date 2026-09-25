"""Shared, dependency-light contracts for the MetaWorld MT50 release runner."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from path_opd.release_contract import (
    ReleaseContractError,
    canonical_json_bytes,
    git_revision,
    tree_sha256,
    validate_checkpoint_manifest,
)


@dataclass(frozen=True)
class Protocol:
    """The fixed MetaWorld protocol used by the reported experiments."""

    suite: str = "metaworld_mt50"
    config_name: str = "pi05_metaworld"
    world_size: int = 8
    student_steps: int = 5
    teacher_steps: int = 5
    model_horizon: int = 5
    execution_prefix: int = 5
    physical_action_dims: int = 4
    model_action_dims: int = 32
    target_chunks: int = 80_000
    updates: int = 10_000
    episode_limit: int = 160
    task_count: int = 50
    variants_per_task: int = 10
    reset_settle_steps: int = 15
    camera_id: int = 2
    rank_seed_multiplier: int = 100_003

    @property
    def evaluation_rows(self) -> int:
        return self.task_count * self.variants_per_task

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PROTOCOL = Protocol()


def paper_contract() -> dict[str, Any]:
    """Return the immutable training contract shared by train and evaluate.

    Keeping this in the dependency-light module prevents the evaluator from
    importing the training worker merely to validate a checkpoint.  The
    checkpoint sidecar is therefore checked against exactly the same contract
    that the training worker writes.
    """

    return {
        "benchmark": PROTOCOL.suite,
        "protocol": PROTOCOL.as_dict(),
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 7.91e-6,
            "betas": [0.9, 0.95],
            "epsilon": 1e-5,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
        },
        "supervision": {
            "path_states": "pre_transition",
            "exclude_endpoint": True,
            "detach_student_path": True,
            "teacher_same_states_and_times": True,
            "teacher_frozen": True,
            "trainable_scope": "action_expert_only",
        },
    }


@dataclass(frozen=True)
class EvaluatorProvenance:
    """Provenance marker carried by every release evaluation artifact."""

    implementation: str = "reconstructed_evaluator"
    byte_identical_to_historical_evaluator: bool = False
    historical_execution_sha256: str = "e97a072798e6d0a24027a6155b57b02806ab895acf071949f498d25888"
    later_reference_sha256: str = "2683ab8d8496754220afbc4e0b6a8de58dc5f318b0ece466a716ed3322bdee1"
    reconstruction_note: str = (
        "Portable reconstruction of the matched MT50 evaluator; the historical "
        "execution file is identified by hash but is not shipped byte-for-byte."
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


EVALUATOR_PROVENANCE = EvaluatorProvenance()

_RLINF_REQUIRED_FILES = (
    "rlinf/models/embodiment/base_policy.py",
    "rlinf/models/embodiment/openpi/__init__.py",
)
_TASK_CONFIG_CANDIDATES = (
    "rlinf/envs/metaworld/metaworld_config.json",
    "rlinf/envs/sim/metaworld/metaworld_config.json",
)


@dataclass(frozen=True)
class RuntimePaths:
    """Validated paths supplied by the caller, with no machine defaults."""

    rlinf_checkout: Path
    norm_stats: Path
    output: Path
    task_config: Path
    base_model: Path | None = None
    teacher_model: Path | None = None
    checkpoint: Path | None = None

    def public_report(self) -> dict[str, str | None]:
        return {
            "rlinf_checkout": str(self.rlinf_checkout),
            "base_model": None if self.base_model is None else str(self.base_model),
            "teacher_model": (None if self.teacher_model is None else str(self.teacher_model)),
            "checkpoint": None if self.checkpoint is None else str(self.checkpoint),
            "norm_stats": str(self.norm_stats),
            "output": str(self.output),
            "task_config": str(self.task_config),
        }


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _require_file(path: Path, label: str) -> Path:
    path = _resolved(path)
    if not path.is_file():
        raise ValueError(f"{label} is not a readable file: {path}")
    return path


def _model_weight_files(model_dir: Path) -> tuple[Path, ...]:
    candidates = list(model_dir.glob("*.safetensors"))
    candidates.extend(
        candidate
        for candidate in (
            model_dir / "model_state_dict/full_weights.pt",
            model_dir / "actor/model_state_dict/full_weights.pt",
        )
        if candidate.is_file()
    )
    return tuple(sorted({path.resolve() for path in candidates}))


def _require_model_directory(path: Path, label: str) -> Path:
    path = _resolved(path)
    if not path.is_dir():
        raise ValueError(f"{label} is not a model directory: {path}")
    if not _model_weight_files(path):
        raise ValueError(
            f"{label} has no RLinf/OpenPI weights (.safetensors or full_weights.pt): {path}"
        )
    return path


def locate_task_config(rlinf_checkout: Path) -> Path:
    """Locate either supported RLinf MetaWorld layout without scanning the tree."""
    for relative in _TASK_CONFIG_CANDIDATES:
        candidate = rlinf_checkout / relative
        if candidate.is_file():
            return candidate.resolve()
    expected = ", ".join(_TASK_CONFIG_CANDIDATES)
    raise ValueError(f"RLinf checkout has no MetaWorld task config; expected one of: {expected}")


def _task_config_path(rlinf_checkout: Path, *, require_existing: bool) -> Path:
    """Resolve a supported task-config layout for real and declaration-only runs.

    Dry-run validation must remain usable before external assets are complete,
    but it should still report an existing ``envs/sim`` layout instead of
    inventing the legacy path.  When no checkout exists yet, retain the stable
    legacy placeholder so declaration-only reports remain deterministic.
    """

    if require_existing:
        return locate_task_config(rlinf_checkout)
    for relative in _TASK_CONFIG_CANDIDATES:
        candidate = rlinf_checkout / relative
        if candidate.is_file():
            return candidate.resolve()
    return rlinf_checkout / _TASK_CONFIG_CANDIDATES[0]


def _validate_output(output: Path, protected: tuple[Path, ...]) -> Path:
    output = _resolved(output)
    if output.exists() and not output.is_dir():
        raise ValueError(f"output must be a directory path: {output}")
    for source in protected:
        if output == source or _is_within(output, source):
            raise ValueError(f"output must be outside input source/model directory: {source}")

    ancestor = output
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
        raise ValueError(f"output has no writable existing ancestor: {output}")
    return output


def validate_train_paths(
    *,
    rlinf_checkout: Path,
    base_model: Path,
    teacher_model: Path,
    norm_stats: Path,
    output: Path,
    require_existing: bool = True,
) -> RuntimePaths:
    """Validate real inputs, or resolve a declaration-only dry-run plan."""
    checkout = _resolved(rlinf_checkout)
    if require_existing and not checkout.is_dir():
        raise ValueError(f"rlinf-checkout is not a directory: {checkout}")
    if require_existing:
        missing = [
            relative for relative in _RLINF_REQUIRED_FILES if not (checkout / relative).is_file()
        ]
        if missing:
            raise ValueError(f"rlinf-checkout is missing required files: {', '.join(missing)}")
    base = (
        _require_model_directory(base_model, "base-model")
        if require_existing
        else _resolved(base_model)
    )
    teacher = (
        _require_model_directory(teacher_model, "teacher-model")
        if require_existing
        else _resolved(teacher_model)
    )
    normalizer = (
        _require_file(norm_stats, "norm-stats")
        if require_existing
        else _resolved(norm_stats)
    )
    destination = _validate_output(output, (checkout, base, teacher))
    return RuntimePaths(
        rlinf_checkout=checkout,
        base_model=base,
        teacher_model=teacher,
        norm_stats=normalizer,
        output=destination,
        task_config=_task_config_path(checkout, require_existing=require_existing),
    )


def validate_evaluate_paths(
    *,
    rlinf_checkout: Path,
    checkpoint: Path,
    norm_stats: Path,
    output: Path,
    require_existing: bool = True,
) -> RuntimePaths:
    """Validate real evaluator inputs, or resolve declaration-only dry-run paths."""
    checkout = _resolved(rlinf_checkout)
    if require_existing and not checkout.is_dir():
        raise ValueError(f"rlinf-checkout is not a directory: {checkout}")
    if require_existing:
        missing = [
            relative for relative in _RLINF_REQUIRED_FILES if not (checkout / relative).is_file()
        ]
        if missing:
            raise ValueError(f"rlinf-checkout is missing required files: {', '.join(missing)}")
    model = (
        _require_model_directory(checkpoint, "checkpoint")
        if require_existing
        else _resolved(checkpoint)
    )
    normalizer = (
        _require_file(norm_stats, "norm-stats")
        if require_existing
        else _resolved(norm_stats)
    )
    destination = _validate_output(output, (checkout, model))
    return RuntimePaths(
        rlinf_checkout=checkout,
        checkpoint=model,
        norm_stats=normalizer,
        output=destination,
        task_config=_task_config_path(checkout, require_existing=require_existing),
    )


def validate_task_config(path: Path) -> list[tuple[str, str]]:
    """Load the ordered MT50 task-to-prompt mapping and enforce all 50 entries."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read MetaWorld task config {path}: {error}") from error
    descriptions = payload.get("TASK_DESCRIPTIONS")
    if not isinstance(descriptions, dict):
        raise ValueError("MetaWorld task config lacks a TASK_DESCRIPTIONS object")
    items: list[tuple[str, str]] = []
    for env_name, prompt in descriptions.items():
        if not isinstance(env_name, str) or not env_name:
            raise ValueError("MetaWorld task names must be non-empty strings")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"MetaWorld prompt for {env_name!r} must be non-empty")
        items.append((env_name, prompt))
    if len(items) != PROTOCOL.task_count:
        raise ValueError(f"expected {PROTOCOL.task_count} MetaWorld tasks, found {len(items)}")
    return items


def activate_rlinf(checkout: Path) -> None:
    """Put the caller-selected RLinf checkout first and reject import drift."""
    checkout = checkout.resolve()
    existing = sys.modules.get("rlinf")
    if existing is not None:
        module_file = getattr(existing, "__file__", None)
        if module_file is None or not _is_within(Path(module_file).resolve(), checkout):
            raise RuntimeError("rlinf was already imported from a different checkout")
    checkout_text = str(checkout)
    if not sys.path or sys.path[0] != checkout_text:
        sys.path.insert(0, checkout_text)
    importlib.invalidate_caches()
    imported = importlib.import_module("rlinf")
    module_file = getattr(imported, "__file__", None)
    if module_file is None or not _is_within(Path(module_file).resolve(), checkout):
        raise RuntimeError("the selected RLinf checkout did not win Python import resolution")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_weight_file(checkpoint: Path) -> Path:
    """Return the unambiguous release checkpoint weights file."""
    files = _model_weight_files(checkpoint)
    exact = checkpoint / "model.safetensors"
    if exact.is_file():
        return exact.resolve()
    if len(files) != 1:
        raise ValueError(
            "checkpoint must contain model.safetensors or exactly one supported weight file"
        )
    return files[0]


def load_checkpoint_manifest(checkpoint: Path) -> dict[str, Any]:
    """Load the JSON sidecar emitted next to a release checkpoint.

    This deliberately does not use ``torch.load``.  Evaluation can therefore
    reject an incompatible checkpoint before importing CUDA, MetaWorld, or
    RLinf, and a malformed sidecar cannot be silently treated as an old
    checkpoint format.
    """

    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"checkpoint manifest.json is required: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read checkpoint manifest {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("checkpoint manifest must contain a JSON object")
    return payload


def _sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"checkpoint {label} must be a 64-character SHA-256 digest")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"checkpoint {label} must be hexadecimal")
    return normalized


def validate_evaluation_checkpoint(
    paths: RuntimePaths,
    *,
    method: str,
    training_seed: int,
) -> dict[str, Any]:
    """Validate an evaluator checkpoint and return a path-free identity report.

    The evaluator can observe the normalization file, task configuration, and
    model weights.  Base-model and teacher bytes are intentionally not CLI
    inputs, so their hashes are required to be present in the release
    manifest but are reported as ``unverifiable`` rather than guessed.  This
    keeps the check fail-closed for missing provenance while avoiding a hidden
    machine path or an implicit model download.
    """

    if paths.checkpoint is None:
        raise ValueError("checkpoint is required for release-contract validation")
    checkpoint = paths.checkpoint
    manifest = load_checkpoint_manifest(checkpoint)
    release = manifest.get("release_contract")
    if not isinstance(release, Mapping):
        raise ValueError("checkpoint manifest lacks a release_contract object")

    # Validate the fields that are duplicated in the human-readable sidecar.
    expected_top_level = {
        "schema_version": 1,
        "method": method,
        "seed": training_seed,
        "world_size": PROTOCOL.world_size,
        "config_name": PROTOCOL.config_name,
        "model_horizon": PROTOCOL.model_horizon,
        "execution_prefix": PROTOCOL.execution_prefix,
        "physical_action_dims": PROTOCOL.physical_action_dims,
        "model_action_dims": PROTOCOL.model_action_dims,
        "student_steps": PROTOCOL.student_steps,
        "teacher_steps": PROTOCOL.teacher_steps,
        "episode_limit": PROTOCOL.episode_limit,
    }
    for key, expected in expected_top_level.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"checkpoint manifest {key} mismatch: {manifest.get(key)!r} != {expected!r}"
            )

    expected_assets = {"base_model", "frozen_teacher", "normalization_stats", "rlinf_task_config"}
    recorded_assets = release.get("asset_sha256")
    if not isinstance(recorded_assets, Mapping):
        raise ValueError("checkpoint release contract lacks asset_sha256")
    for name in expected_assets:
        if name not in recorded_assets:
            raise ValueError(f"checkpoint release contract lacks asset {name}")
        _sha256_text(recorded_assets[name], f"asset {name}")

    recorded_code = release.get("code")
    if not isinstance(recorded_code, Mapping):
        raise ValueError("checkpoint release contract lacks code provenance")
    if "release_source_sha256" not in recorded_code or "rlinf_revision" not in recorded_code:
        raise ValueError("checkpoint release contract lacks complete code provenance")
    _sha256_text(recorded_code["release_source_sha256"], "code release_source_sha256")
    revision = recorded_code["rlinf_revision"]
    if revision is not None and (
        not isinstance(revision, str)
        or len(revision) not in (40, 64)
        or any(character not in "0123456789abcdefABCDEF" for character in revision)
    ):
        raise ValueError("checkpoint code rlinf_revision is not a git revision")
    if "panel" not in release:
        raise ValueError("checkpoint release contract lacks panel field")
    sidecar_update = manifest.get("optimizer_update")
    if (
        not isinstance(sidecar_update, int)
        or isinstance(sidecar_update, bool)
        or sidecar_update < 0
    ):
        raise ValueError("checkpoint manifest optimizer_update must be a non-negative integer")
    release_update = release.get("optimizer_update")
    if (
        not isinstance(release_update, int)
        or isinstance(release_update, bool)
        or release_update < 0
    ):
        raise ValueError("checkpoint release optimizer_update must be a non-negative integer")

    try:
        weight_path = checkpoint_weight_file(checkpoint)
        normalizer_sha256 = file_sha256(paths.norm_stats)
        model_file_sha256 = file_sha256(weight_path)
        normalizer_tree_sha256 = tree_sha256(paths.norm_stats)
        task_config_tree_sha256 = tree_sha256(paths.task_config)
        code_hashes = {
            "release_source_sha256": tree_sha256(
                Path(__file__).resolve().parents[2], include=("src", "benchmarks", "configs")
            ),
            "rlinf_revision": git_revision(paths.rlinf_checkout),
        }
        validate_checkpoint_manifest(
            release,
            expected={
                "benchmark": PROTOCOL.suite,
                "method": method,
                "seed": training_seed,
                "world_size": PROTOCOL.world_size,
                "contract": paper_contract(),
            },
            assets={
                "normalization_stats": normalizer_tree_sha256,
                "rlinf_task_config": task_config_tree_sha256,
            },
            code=code_hashes,
        )
    except (OSError, ReleaseContractError, ValueError) as error:
        raise ValueError(f"checkpoint release contract mismatch: {error}") from error

    if manifest.get("normalizer_sha256") != normalizer_sha256:
        raise ValueError("checkpoint manifest normalizer_sha256 does not match --norm")
    if manifest.get("model_file_sha256") != model_file_sha256:
        raise ValueError("checkpoint manifest model_file_sha256 does not match checkpoint weights")
    if release_update != sidecar_update:
        raise ValueError("checkpoint release optimizer_update disagrees with sidecar")

    release_digest = hashlib.sha256(canonical_json_bytes(dict(release))).hexdigest()
    return {
        "verified": True,
        "manifest_schema_version": manifest["schema_version"],
        "release_schema": release["schema"],
        "release_contract_schema": release["contract_schema"],
        "method": method,
        "training_seed": training_seed,
        "world_size": PROTOCOL.world_size,
        "model_file_sha256": model_file_sha256,
        "normalizer_sha256": normalizer_sha256,
        "normalization_stats_tree_sha256": normalizer_tree_sha256,
        "rlinf_task_config_tree_sha256": task_config_tree_sha256,
        "code_hashes_verified": sorted(code_hashes),
        "release_contract_sha256": release_digest,
        "unverifiable_assets": ["base_model", "frozen_teacher"],
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a small JSON status/summary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def dry_run_plan(
    operation: str,
    paths: RuntimePaths,
    *,
    method: str | None = None,
    seed: int | None = None,
    target_chunks: int | None = None,
    max_updates: int | None = None,
) -> dict[str, Any]:
    """Create an auditable plan that cannot be mistaken for an executed run."""
    task_config_present = paths.task_config.is_file()
    tasks = validate_task_config(paths.task_config) if task_config_present else []
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "DRY_RUN_ONLY",
        "execution_performed": False,
        "operation": operation,
        "protocol": PROTOCOL.as_dict(),
        "paths": paths.public_report(),
        "checks": {
            "paths_exist": all(
                path.exists()
                for path in (
                    paths.rlinf_checkout,
                    paths.norm_stats,
                    paths.output,
                    paths.task_config,
                    *(
                        item
                        for item in (paths.base_model, paths.teacher_model, paths.checkpoint)
                        if item
                    ),
                )
            ),
            "rlinf_contract_files_present": all(
                (paths.rlinf_checkout / relative).is_file() for relative in _RLINF_REQUIRED_FILES
            ),
            "model_weights_present": bool(
                paths.base_model
                and paths.teacher_model
                and _model_weight_files(paths.base_model)
                and _model_weight_files(paths.teacher_model)
            )
            if operation == "train"
            else bool(paths.checkpoint and _model_weight_files(paths.checkpoint)),
            "normalizer_present": paths.norm_stats.is_file(),
            "task_count": len(tasks) if task_config_present else PROTOCOL.task_count,
            "task_config_validated": task_config_present,
            "heavy_runtime_imported": False,
            "gpu_work_performed": False,
            "simulator_work_performed": False,
        },
    }
    if operation == "train":
        if method not in {"path_opd", "endpoint_dagger"}:
            raise ValueError(f"unsupported training method: {method!r}")
        chunks = PROTOCOL.target_chunks if target_chunks is None else target_chunks
        if chunks <= 0 or chunks % PROTOCOL.world_size:
            raise ValueError("target-chunks must be a positive multiple of 8")
        planned_updates = chunks // PROTOCOL.world_size
        if max_updates is not None and not 1 <= max_updates <= planned_updates:
            raise ValueError("max-updates must be between 1 and target-chunks / 8")
        effective_updates = planned_updates if max_updates is None else max_updates
        if max_updates is not None:
            run_scope = "qualification"
        elif chunks == PROTOCOL.target_chunks:
            run_scope = "paper"
        else:
            run_scope = "custom_budget"
        payload["training"] = {
            "method": method,
            "seed": seed,
            "target_chunks": chunks,
            "planned_updates": planned_updates,
            "effective_updates": effective_updates,
            "effective_chunks": effective_updates * PROTOCOL.world_size,
            "run_scope": run_scope,
            "launcher": "torchrun",
            "backend": "nccl",
        }
    elif operation == "evaluate":
        payload["evaluation"] = {
            "seed": seed,
            "expected_rows": PROTOCOL.evaluation_rows,
            "matched_variants": True,
            "resume_from_rows_jsonl": True,
            "provenance": EVALUATOR_PROVENANCE.as_dict(),
        }
    else:
        raise ValueError(f"unsupported operation: {operation!r}")
    return payload

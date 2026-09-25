"""Dependency-light CALVIN ABC-D protocol and input contracts.

This module intentionally imports only the Python standard library.  It is the
boundary between a reviewable release and the optional CALVIN/RLinf runtime.
No path in this file is inferred from the machine that produced the paper
results; all model, simulator, annotation, and panel locations are caller
parameters.
"""

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


class ContractError(ValueError):
    """Raised when a run cannot be identified as the declared protocol."""


@dataclass(frozen=True)
class Protocol:
    benchmark: str = "calvin_abc_d"
    task_suite: str = "calvin_abc"
    evaluation_scene: str = "calvin_scene_D"
    config_name: str = "pi05_calvin"
    world_size: int = 8
    student_steps: int = 8
    teacher_steps: int = 8
    model_horizon: int = 5
    execution_prefix: int = 5
    physical_action_dims: int = 7
    model_action_dims: int = 32
    target_chunks: int = 80_000
    optimizer_updates: int = 10_000
    training_episode_limit: int = 480
    evaluation_subtask_limit: int = 360
    sequence_length: int = 5
    evaluation_rows: int = 1_000
    action_repeat: int = 8
    controller_frequency_hz: int = 30
    simulation_frequency_hz: int = 240
    rank_seed_multiplier: int = 1
    checkpoint_chunks: tuple[int, ...] = (10_000, 20_000, 40_000, 60_000, 80_000)
    reset_interval_chunks: int = 16

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["checkpoint_chunks"] = list(self.checkpoint_chunks)
        return result


PROTOCOL = Protocol()

# ``clip_grad`` is the one public spelling.  The historical worker once read
# ``clip`` while the source contract emitted ``clip_grad``; accepting both at
# runtime would make a silent optimizer change possible, so the release rejects
# the legacy alias instead.
OPTIMIZER: dict[str, Any] = {
    "name": "AdamW",
    "learning_rate": 7.91e-6,
    "betas": [0.9, 0.95],
    "epsilon": 1e-5,
    "weight_decay": 0.01,
    "clip_grad": 1.0,
}

FLOW_TIMES = tuple(1.0 - index / PROTOCOL.student_steps for index in range(PROTOCOL.student_steps))
TIME_WEIGHTS = tuple(1.0 / PROTOCOL.student_steps for _ in FLOW_TIMES)
SAFE_BOUNDARY = "post_update_pre_selection_reset_noise"
PANEL_SHA256 = "0aaa7c37dd5976b3f4372b8501e67e692ff565ba48730c0481aa3f92ff71ff60"
SCHEMA_VERSION = 1


def resolve_clip(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve and validate the single global-L2 clipping field."""

    values = OPTIMIZER if config is None else config
    if "clip" in values:
        raise ContractError("optimizer key 'clip' is not supported; use 'clip_grad'")
    value = values.get("clip_grad")
    if isinstance(value, bool) or not isinstance(value, int | float) or float(value) <= 0:
        raise ContractError("optimizer.clip_grad must be a positive number")
    return {
        "mode": "global_norm",
        "norm_type": 2.0,
        "max_norm": float(value),
        "source_key": "clip_grad",
    }


CLIP_RESOLUTION = resolve_clip()


def paper_contract() -> dict[str, Any]:
    """Return the JSON-safe protocol used by dry-run and checkpoint reports."""

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL.as_dict(),
        "optimizer": {**OPTIMIZER, "clip_resolution": CLIP_RESOLUTION},
        "flow": {
            "solver": "euler",
            "student_steps": PROTOCOL.student_steps,
            "teacher_steps": PROTOCOL.teacher_steps,
            "pre_transition_times": list(FLOW_TIMES),
            "time_weights": list(TIME_WEIGHTS),
        },
        "supervision": {
            "student_path_states": "pre_transition",
            "exclude_endpoint": True,
            "detach_student_path": True,
            "teacher_query": "same_student_states_and_times",
            "teacher_frozen": True,
            "teacher_actor_only": True,
            "trainable_scope": "action_expert_only",
            "dagger_target": "teacher_rollout_endpoint",
            "dagger_teacher_rollout": "one endpoint rollout with eight velocity NFE",
        },
        "evaluation": {
            "protocol": "official_d",
            "rows": PROTOCOL.evaluation_rows,
            "sequence_length": PROTOCOL.sequence_length,
            "subtask_limit_primitive_steps": PROTOCOL.evaluation_subtask_limit,
            "metrics": ["average_completed_sequence_length", "SR1", "SR2", "SR3", "SR4", "SR5"],
            "panel_sha256": PANEL_SHA256,
        },
    }


def _resolved(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_output(path: Path, protected: tuple[Path, ...], *, require_writable: bool) -> Path:
    path = _resolved(path)
    for source in protected:
        if path == source or _within(path, source):
            raise ContractError(f"output must be outside input path: {source}")
    if path.exists() and not path.is_dir():
        raise ContractError(f"output must be a directory: {path}")
    if require_writable:
        ancestor = path
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
            raise ContractError(f"output has no writable existing ancestor: {path}")
    return path


def _require(path: Path | None, label: str, *, directory: bool, required: bool) -> Path | None:
    if path is None:
        if required:
            raise ContractError(f"{label} must be supplied explicitly")
        return None
    path = _resolved(path)
    if required and (
        not path.exists()
        or (directory and not path.is_dir())
        or (not directory and not path.is_file())
    ):
        kind = "directory" if directory else "file"
        raise ContractError(f"{label} is not a readable {kind}: {path}")
    return path


def _model_dir(path: Path | None, label: str, *, required: bool) -> Path | None:
    value = _require(path, label, directory=True, required=required)
    if value is None or not required:
        return value
    candidates = (
        value / "model.safetensors",
        value / "model_state_dict" / "full_weights.pt",
        value / "actor" / "model_state_dict" / "full_weights.pt",
    )
    if not any(candidate.is_file() for candidate in candidates):
        raise ContractError(f"{label} has no supported model weights: {value}")
    return value


@dataclass(frozen=True)
class RuntimePaths:
    """All external inputs for one CALVIN operation."""

    rlinf_checkout: Path
    normalization_stats: Path
    environment_assets: Path
    task_oracle_annotations: Path
    output: Path
    base_model: Path | None = None
    frozen_teacher: Path | None = None
    checkpoint: Path | None = None
    official_d_panel: Path | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {key: (None if value is None else str(value)) for key, value in asdict(self).items()}


_RLINF_REQUIRED = (
    "rlinf/models/embodiment/base_policy.py",
    "rlinf/models/embodiment/openpi/__init__.py",
    "rlinf/envs/calvin/__init__.py",
)


def validate_checkout(path: Path, *, required: bool) -> Path:
    path = _resolved(path)
    if required:
        if not path.is_dir():
            raise ContractError(f"rlinf-checkout is not a directory: {path}")
        missing = [relative for relative in _RLINF_REQUIRED if not (path / relative).is_file()]
        if missing:
            raise ContractError("rlinf-checkout is missing: " + ", ".join(missing))
    return path


def validate_train_paths(
    *,
    rlinf_checkout: Path,
    base_model: Path,
    frozen_teacher: Path,
    normalization_stats: Path,
    environment_assets: Path,
    task_oracle_annotations: Path,
    output: Path,
    require_existing: bool,
) -> RuntimePaths:
    checkout = validate_checkout(rlinf_checkout, required=require_existing)
    base = _model_dir(base_model, "base-model", required=require_existing)
    teacher = _model_dir(frozen_teacher, "frozen-teacher", required=require_existing)
    # Dry-run validates the declared path shape without requiring external
    # assets to be mounted.  A real run still requires the stats file.
    norm = _require(
        normalization_stats,
        "normalization-stats",
        directory=False,
        required=require_existing,
    )
    env_assets = _require(
        environment_assets, "environment-assets", directory=True, required=require_existing
    )
    oracle = _require(
        task_oracle_annotations,
        "task-oracle-annotations",
        directory=False,
        required=require_existing,
    )
    destination = _validate_output(
        output,
        tuple(item for item in (checkout, base, teacher) if item),
        require_writable=require_existing,
    )
    return RuntimePaths(
        checkout, norm, env_assets, oracle, destination, base_model=base, frozen_teacher=teacher
    )


def validate_evaluate_paths(
    *,
    rlinf_checkout: Path,
    base_model: Path,
    checkpoint: Path,
    normalization_stats: Path,
    environment_assets: Path,
    task_oracle_annotations: Path,
    official_d_panel: Path,
    output: Path,
    require_existing: bool,
    allow_custom_panel: bool = False,
) -> RuntimePaths:
    checkpoint_path = _resolved(checkpoint)
    if require_existing and not checkpoint_path.exists():
        raise ContractError(f"checkpoint does not exist: {checkpoint_path}")
    # Validate the checkpoint contract before unrelated external runtime
    # directories so a missing manifest is reported deterministically.
    if require_existing:
        checkpoint_artifact(checkpoint_path)
        load_checkpoint_manifest(checkpoint_path)
    checkout = validate_checkout(rlinf_checkout, required=require_existing)
    base = _model_dir(base_model, "base-model", required=require_existing)
    # Keep dry-run dependency-free: callers may inspect the contract before
    # downloading the normalizer.  Evaluation enforces the file at runtime.
    norm = _require(
        normalization_stats,
        "normalization-stats",
        directory=False,
        required=require_existing,
    )
    env_assets = _require(
        environment_assets, "environment-assets", directory=True, required=require_existing
    )
    oracle = _require(
        task_oracle_annotations,
        "task-oracle-annotations",
        directory=False,
        required=require_existing,
    )
    panel = _require(
        official_d_panel, "official-d-panel", directory=False, required=require_existing
    )
    if require_existing and panel is not None:
        validate_panel(panel, allow_custom=allow_custom_panel)
    destination = _validate_output(
        output,
        tuple(item for item in (checkout, base, checkpoint_path) if item),
        require_writable=require_existing,
    )
    return RuntimePaths(
        checkout,
        norm,
        env_assets,
        oracle,
        destination,
        base_model=base,
        checkpoint=checkpoint_path,
        official_d_panel=panel,
    )


def activate_rlinf(checkout: Path) -> Any:
    """Put the explicitly selected RLinf checkout first in import resolution."""

    checkout = _resolved(checkout)
    module = sys.modules.get("rlinf")
    if module is not None:
        module_path = getattr(module, "__file__", None)
        if module_path is None or not _within(_resolved(module_path), checkout):
            raise RuntimeError("rlinf was already imported from a different checkout")
    text = str(checkout)
    if not sys.path or sys.path[0] != text:
        sys.path.insert(0, text)
    importlib.invalidate_caches()
    loaded = importlib.import_module("rlinf")
    loaded_path = getattr(loaded, "__file__", None)
    if loaded_path is None or not _within(_resolved(loaded_path), checkout):
        raise RuntimeError("selected RLinf checkout did not win import resolution")
    return loaded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    # Checkpoint manifests are portable release artifacts; never serialize a
    # caller's absolute filesystem location.
    item: dict[str, Any] = {
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }
    if path.is_file():
        item["bytes"] = path.stat().st_size
        item["sha256"] = sha256_file(path)
    return item


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode(
        "utf-8"
    )


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_panel(path: Path, *, allow_custom: bool = False) -> dict[str, Any]:
    """Validate the fixed Official-D row identity and return digest evidence."""

    path = _resolved(path)
    if not path.is_file():
        raise ContractError(f"official-D panel is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"could not read official-D panel: {error}") from error
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        count = None
        raise ContractError(
            f"official-D panel must contain {PROTOCOL.evaluation_rows} rows, got {count}"
        )
    if not allow_custom and len(rows) != PROTOCOL.evaluation_rows:
        count = len(rows)
        raise ContractError(
            f"official-D panel must contain {PROTOCOL.evaluation_rows} rows, got {count}"
        )
    if allow_custom and not 1 <= len(rows) <= PROTOCOL.evaluation_rows:
        count = len(rows)
        raise ContractError(
            "custom Official-D panel must contain between 1 and "
            f"{PROTOCOL.evaluation_rows} rows, got {count}"
        )
    identities: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ContractError(f"panel row {index} is not an object")
        if row.get("sequence_index") != index:
            raise ContractError(f"panel sequence_index must be contiguous at row {index}")
        identity = row.get("identity_sha256")
        if not _valid_digest(identity):
            raise ContractError(f"panel row {index} has invalid identity_sha256")
        if not isinstance(row.get("initial_state"), Mapping):
            raise ContractError(f"panel row {index} lacks initial_state mapping")
        subtasks = row.get("subtasks")
        if (
            not isinstance(subtasks, list)
            or len(subtasks) != PROTOCOL.sequence_length
            or not all(isinstance(item, str) and item for item in subtasks)
        ):
            raise ContractError(f"panel row {index} must contain five non-empty subtasks")
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ContractError("official-D panel contains duplicate identities")
    digest = sha256_file(path)
    if not allow_custom and digest != PANEL_SHA256:
        raise ContractError(
            "official-D panel digest is not the recorded formal panel; "
            "pass --allow-custom-panel for a qualification panel"
        )
    return {
        "path": str(path),
        "sha256": digest,
        "formal_panel": digest == PANEL_SHA256 and len(rows) == PROTOCOL.evaluation_rows,
        "row_count": len(rows),
        "ordered_identity_sha256": hashlib.sha256(canonical_json(identities)).hexdigest(),
        "rows": rows,
    }


def rows_for_rank(panel: Mapping[str, Any], rank: int, world_size: int) -> list[dict[str, Any]]:
    if world_size < 1 or not 0 <= rank < world_size:
        raise ContractError("rank/world_size is invalid")
    rows = list(panel["rows"])
    if world_size == PROTOCOL.world_size:
        if len(rows) % world_size != 0:
            raise ContractError(
                "custom Official-D panel row count must be divisible by the launched world size"
            )
        assigned = [row for row in rows if int(row["sequence_index"]) % world_size == rank]
        expected = len(rows) // world_size
        if len(assigned) != expected:
            raise ContractError(
                f"rank {rank} does not own the expected {expected} Official-D rows"
            )
        return assigned
    if world_size == 1:
        return rows
    raise ContractError("formal Official-D evaluation requires exactly eight ranks")


def checkpoint_artifact(path: Path) -> Path:
    path = _resolved(path)
    if path.is_file():
        return path
    for name in ("trainer_state.pt", "checkpoint.pt"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise ContractError(
        f"checkpoint must be a file or contain trainer_state.pt/checkpoint.pt: {path}"
    )


def checkpoint_manifest_path(path: Path) -> Path:
    path = _resolved(path)
    if path.is_dir():
        return path / "manifest.json"
    return path.with_name("manifest.json")


def load_checkpoint_manifest(path: Path) -> dict[str, Any]:
    manifest_path = checkpoint_manifest_path(path)
    if not manifest_path.is_file():
        raise ContractError(f"checkpoint manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid checkpoint manifest: {error}") from error
    if not isinstance(payload, dict):
        raise ContractError("checkpoint manifest must be an object")
    return payload


def validate_checkpoint_manifest(
    manifest: Mapping[str, Any],
    *,
    method: str,
    seed: int,
    world_size: int,
    normalizer_sha256: str | None = None,
    expected_teacher_sha256: str | None = None,
    release_contract: Mapping[str, Any] | None = None,
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": PROTOCOL.benchmark,
        "method": method,
        "seed": seed,
        "world_size": world_size,
        "student_steps": PROTOCOL.student_steps,
        "model_horizon": PROTOCOL.model_horizon,
        "execution_prefix": PROTOCOL.execution_prefix,
        "physical_action_dims": PROTOCOL.physical_action_dims,
        "safe_boundary": SAFE_BOUNDARY,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ContractError(f"checkpoint manifest does not match declared contract: {mismatches}")
    if (
        normalizer_sha256 is not None
        and manifest.get("normalization_stats_sha256") != normalizer_sha256
    ):
        raise ContractError("checkpoint uses different normalization statistics")
    if (
        expected_teacher_sha256 is not None
        and manifest.get("teacher_sha256") != expected_teacher_sha256
    ):
        raise ContractError("checkpoint uses a different frozen teacher")
    if release_contract is not None:
        recorded = manifest.get("release_contract")
        if recorded != release_contract:
            raise ContractError("checkpoint release contract does not match this run")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def dry_run_plan(
    operation: str,
    paths: RuntimePaths,
    *,
    method: str | None,
    seed: int,
    target_chunks: int,
    max_updates: int | None,
    panel: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if operation not in {"train", "evaluate"}:
        raise ContractError(f"unsupported operation: {operation}")
    if method is not None and method not in {"path_opd", "endpoint_dagger"}:
        raise ContractError(f"unsupported training method: {method}")
    if seed not in (0, 1, 2):
        raise ContractError("seed must be one of 0, 1, 2")
    if target_chunks <= 0 or target_chunks % PROTOCOL.world_size:
        raise ContractError("target-chunks must be a positive multiple of eight")
    planned_updates = target_chunks // PROTOCOL.world_size
    if max_updates is not None and not 1 <= max_updates <= planned_updates:
        raise ContractError("max-updates must be between one and target updates")
    payload: dict[str, Any] = {
        "schema": f"path-opd-calvin-{operation}-v1",
        "status": "DRY_RUN_ONLY",
        "execution_performed": False,
        "paper_contract": paper_contract(),
        "paths": paths.as_dict(),
        "checks": {
            "heavy_runtime_imported": False,
            "gpu_work_performed": False,
            "simulator_work_performed": False,
            "private_paths_inferred": False,
        },
    }
    if operation == "train":
        effective = planned_updates if max_updates is None else max_updates
        payload["training"] = {
            "method": method,
            "seed": seed,
            "target_chunks": target_chunks,
            "planned_updates": planned_updates,
            "effective_updates": effective,
            "effective_chunks": effective * PROTOCOL.world_size,
            "run_scope": "paper"
            if max_updates is None and target_chunks == PROTOCOL.target_chunks
            else "qualification",
            "launcher": "torchrun",
            "backend": "nccl",
            "resume_safe_boundary": SAFE_BOUNDARY,
        }
    else:
        payload["evaluation"] = {
            "world_size": PROTOCOL.world_size,
            "expected_rows": PROTOCOL.evaluation_rows,
            "panel": None
            if panel is None
            else {key: panel[key] for key in ("sha256", "formal_panel", "row_count")},
            "protocol": "official_d",
            "outcome_metrics": [
                "average_completed_sequence_length",
                "SR1",
                "SR2",
                "SR3",
                "SR4",
                "SR5",
            ],
        }
    return payload

"""Shared public contract for the ManiSkill Path-OPD release runner.

The training and evaluation implementations are an intentionally small,
parameterized extraction of the archived RLinf/OpenPI workers.  The upstream
source notices are retained in the executable modules; this file contains only
release-side contract and validation code.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised when a caller asks for a non-reproducible benchmark run."""


@dataclass(frozen=True)
class BenchmarkContract:
    benchmark: str = "maniskill"
    task: str = "PutOnPlateInScene25Main-v3"
    method_choices: tuple[str, ...] = ("path_opd", "endpoint_dagger")
    supported_seeds: tuple[int, ...] = (0, 1, 2)
    solver_steps: int = 8
    model_horizon: int = 8
    executed_prefix: int = 5
    physical_action_dims: int = 7
    world_size: int = 8
    per_rank_batch_size: int = 1
    global_batch_size: int = 8
    fresh_chunks: int = 80_000
    optimizer_updates: int = 10_000
    checkpoint_chunks: tuple[int, ...] = (10_000, 20_000, 40_000, 60_000, 80_000)
    episode_primitive_steps: int = 80
    control_frequency_hz: int = 5
    simulation_frequency_hz: int = 500
    evaluation_rows: int = 320
    evaluation_protocol: str = "fixed_ten_point_panel"
    optimizer_name: str = "AdamW"
    learning_rate: float = 7.91e-6
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-5
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0


CONTRACT = BenchmarkContract()

# Keep the simulator action semantics identical between training and evaluation.
# This is the mode selected by RLinf's ``widowx_bridge`` policy setup.
MANISKILL_CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


@dataclass(frozen=True)
class ResolvedContract:
    """Run-specific contract emitted by dry-run and real-run reports."""

    method: str
    seed: int
    world_size: int
    max_updates: int
    local_fresh_chunks: int
    global_fresh_chunks: int
    formal_full_budget: bool
    qualification: bool
    solver_steps: int = CONTRACT.solver_steps
    model_horizon: int = CONTRACT.model_horizon
    executed_prefix: int = CONTRACT.executed_prefix
    physical_action_dims: int = CONTRACT.physical_action_dims
    global_batch_size: int = CONTRACT.global_batch_size
    evaluation_rows: int = CONTRACT.evaluation_rows

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_contract(
    *,
    method: str,
    seed: int,
    world_size: int,
    max_updates: int,
) -> ResolvedContract:
    """Validate and resolve one parameterized training invocation.

    ``max_updates=10_000`` is the paper budget.  A smaller value is a
    qualification/diagnostic run and is never labelled as a paper result.
    ``world_size`` is exposed for local qualification, but the full budget is
    formal only at the recorded eight-rank topology.
    """

    if method not in CONTRACT.method_choices:
        raise ContractError(f"method must be one of {CONTRACT.method_choices!r}")
    if seed not in CONTRACT.supported_seeds:
        raise ContractError(f"seed must be one of {CONTRACT.supported_seeds!r}, got {seed!r}")
    if isinstance(world_size, bool) or world_size < 1:
        raise ContractError("world_size must be a positive integer")
    if isinstance(max_updates, bool) or max_updates < 1:
        raise ContractError("max_updates must be a positive integer")
    if max_updates > CONTRACT.optimizer_updates:
        raise ContractError(
            f"max_updates cannot exceed the paper budget {CONTRACT.optimizer_updates}"
        )
    global_chunks = max_updates * world_size
    formal_full_budget = (
        max_updates == CONTRACT.optimizer_updates and world_size == CONTRACT.world_size
    )
    return ResolvedContract(
        method=method,
        seed=seed,
        world_size=world_size,
        max_updates=max_updates,
        local_fresh_chunks=max_updates,
        global_fresh_chunks=global_chunks,
        formal_full_budget=formal_full_budget,
        qualification=not formal_full_budget,
        global_batch_size=world_size,
    )


def paper_contract_dict() -> dict[str, Any]:
    """Return a JSON-safe description of the fixed ManiSkill protocol."""

    payload = asdict(CONTRACT)
    payload["method_choices"] = list(CONTRACT.method_choices)
    payload["supported_seeds"] = list(CONTRACT.supported_seeds)
    payload["checkpoint_chunks"] = list(CONTRACT.checkpoint_chunks)
    payload["betas"] = list(CONTRACT.betas)
    payload["action_contract"] = {
        "model_horizon": CONTRACT.model_horizon,
        "executed_prefix": CONTRACT.executed_prefix,
        "physical_action_dims": CONTRACT.physical_action_dims,
        "control_mode": MANISKILL_CONTROL_MODE,
    }
    payload["flow"] = {
        "solver": "euler",
        "student_steps": CONTRACT.solver_steps,
        "teacher_steps": CONTRACT.solver_steps,
        "pre_transition_times": [
            1.0 - index / CONTRACT.solver_steps for index in range(CONTRACT.solver_steps)
        ],
        "time_weights": [1.0 / CONTRACT.solver_steps] * CONTRACT.solver_steps,
    }
    payload["supervision"] = {
        "path_states": "student_rollout_chains[:, :-1]",
        "detach_and_clone": True,
        "teacher_query": "same_student_states_and_times",
        "teacher_gradients": False,
        "trainable_scope": "action_expert_only",
        "endpoint_dagger_loss": "native_conditional_flow_matching",
    }
    payload["evaluation"] = {
        "protocol": CONTRACT.evaluation_protocol,
        "rows": CONTRACT.evaluation_rows,
        "primitive_steps": CONTRACT.episode_primitive_steps,
        "metrics": ["success_once", "success_at_end"],
        "fixed_reset_ids": True,
    }
    return payload


ASSET_PARAMETERS_TRAIN = (
    "base_model",
    "frozen_teacher",
    "normalization_stats",
    "teacher_normalization_stats",
    "simulator_assets",
    "maniskill_package_assets",
)
ASSET_PARAMETERS_EVAL = (
    "base_model",
    "normalization_stats",
    "simulator_assets",
    "maniskill_package_assets",
    "evaluation_panel",
)

RLINF_REQUIRED_FILES = (
    "rlinf/envs/action_utils.py",
    "rlinf/envs/maniskill/maniskill_env.py",
    "rlinf/envs/maniskill/maniskill_offload_env.py",
    "rlinf/models/embodiment/base_policy.py",
    "rlinf/models/embodiment/openpi/__init__.py",
)


def _path_value(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    # Do not resolve absent paths: dry-run reports the caller's explicit value
    # without inventing a host-dependent fallback.
    return path


def validate_assets(
    supplied: Mapping[str, str | os.PathLike[str] | None],
    *,
    purpose: str,
    require_existing: bool,
) -> dict[str, Path | None]:
    """Resolve explicit asset arguments and fail closed on missing/extra ones."""

    # The first release exposed one normalizer argument and implicitly reused it
    # for the frozen teacher.  Keep that invocation valid while making the
    # teacher asset an explicit, independently hashable input for new runs.
    supplied = dict(supplied)
    if (
        purpose == "train"
        and "teacher_normalization_stats" not in supplied
        and "normalization_stats" in supplied
    ):
        supplied["teacher_normalization_stats"] = supplied["normalization_stats"]
    expected = ASSET_PARAMETERS_TRAIN if purpose == "train" else ASSET_PARAMETERS_EVAL
    unknown = sorted(set(supplied) - set(expected))
    if unknown:
        raise ContractError(f"unexpected asset parameter(s): {', '.join(unknown)}")
    resolved = {name: _path_value(supplied.get(name)) for name in expected}
    if require_existing:
        missing = [name for name, path in resolved.items() if path is None]
        if missing:
            raise ContractError(
                "real benchmark execution requires explicit asset paths: " + ", ".join(missing)
            )
        absent = [name for name, path in resolved.items() if path is not None and not path.exists()]
        if absent:
            details = ", ".join(f"{name}={resolved[name]}" for name in absent)
            raise ContractError(f"asset path does not exist: {details}")
        for name in ("base_model", "frozen_teacher"):
            model = resolved[name]
            if model is None or not model.is_dir():
                raise ContractError(f"{name} must be a model directory: {model}")
            nested_actor_weights = model / "actor" / "model.safetensors"
            if name == "frozen_teacher" and nested_actor_weights.is_file():
                raise ContractError(
                    "frozen_teacher points to the downloaded model root "
                    f"({model}); weights are under actor/model.safetensors. "
                    f"Pass {model / 'actor'} as --teacher-model."
                )
            weight_files = (
                model / "model.safetensors",
                model / "model_state_dict" / "full_weights.pt",
                model / "actor" / "model_state_dict" / "full_weights.pt",
            )
            if not any(candidate.is_file() for candidate in weight_files):
                raise ContractError(f"{name} has no supported model weights: {model}")
    return resolved


def validate_rlinf_checkout(path: Path, *, require_existing: bool) -> Path:
    """Validate the explicit RLinf checkout before importing any runtime code."""

    checkout = path.expanduser().resolve(strict=False)
    if not require_existing:
        return checkout
    if not checkout.is_dir() or not (checkout / "rlinf").is_dir():
        raise ContractError("RLinf checkout must be a directory containing rlinf/")
    missing = [relative for relative in RLINF_REQUIRED_FILES if not (checkout / relative).is_file()]
    if missing:
        raise ContractError("RLinf checkout is missing: " + ", ".join(missing))
    return checkout


def sha256_file(path: Path) -> str:
    """Hash one explicit file for output provenance (never a private fallback)."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_fingerprint(path: Path) -> dict[str, Any]:
    """Return lightweight explicit-path metadata without recursively scanning it."""

    item: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }
    if path.is_file():
        item["bytes"] = path.stat().st_size
        item["sha256"] = sha256_file(path)
    return item


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def validate_panel(path: Path) -> dict[str, Any]:
    """Validate the public fixed-panel shape before importing simulator code."""

    if not path.is_file():
        raise ContractError(f"evaluation panel is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"could not read evaluation panel {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ContractError("evaluation panel must be an object with a rows array")
    rows = payload["rows"]
    if len(rows) != CONTRACT.evaluation_rows:
        raise ContractError(
            "evaluation panel must contain exactly "
            f"{CONTRACT.evaluation_rows} rows, got {len(rows)}"
        )
    if payload.get("denominator", CONTRACT.evaluation_rows) != CONTRACT.evaluation_rows:
        raise ContractError("evaluation panel denominator must be 320")
    if payload.get("environment_id", CONTRACT.task) != CONTRACT.task:
        raise ContractError(f"evaluation panel environment must be {CONTRACT.task!r}")
    ids: list[int] = []
    owners: set[tuple[int, int, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ContractError(f"evaluation panel row {index} is not an object")
        reset_id = row.get("reset_episode_id", row.get("initial_state_id"))
        if type(reset_id) is not int or reset_id < 0:
            raise ContractError(f"evaluation panel row {index} has invalid reset_episode_id")
        initial_id = row.get("initial_state_id", reset_id)
        if type(initial_id) is not int or initial_id != reset_id:
            raise ContractError(
                f"evaluation panel row {index} reset_episode_id/initial_state_id mismatch"
            )
        owner_values = tuple(row.get(key) for key in ("worker_rank", "pipeline_stage", "env_slot"))
        if all(type(value) is int for value in owner_values):
            if owner_values in owners:
                raise ContractError(f"evaluation panel duplicates owner {owner_values!r}")
            owners.add(owner_values)  # type: ignore[arg-type]
        ids.append(reset_id)
    if len(set(ids)) != len(ids):
        raise ContractError("evaluation panel contains duplicate reset IDs")
    if owners and len(owners) != len(rows):
        raise ContractError("evaluation panel must specify ownership for every row or none")
    if owners:
        for rank in range(CONTRACT.world_size):
            rank_owners = sorted(owner for owner in owners if owner[0] == rank)
            expected_owners = [
                (rank, 0, slot) for slot in range(CONTRACT.evaluation_rows // CONTRACT.world_size)
            ]
            if rank_owners != expected_owners:
                raise ContractError(
                    f"evaluation panel ownership for rank {rank} is not stage 0, slots 0..39"
                )
    result = dict(payload)
    result["rows"] = rows
    result["row_count"] = len(rows)
    result["file_sha256"] = sha256_file(path)
    result["ordered_reset_ids_sha256"] = hashlib.sha256(canonical_json_bytes(ids)).hexdigest()
    content_payload = dict(payload)
    content_payload.pop("panel_sha256", None)
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(content_payload)).hexdigest()
    if payload.get("panel_sha256", result["content_sha256"]) != result["content_sha256"]:
        raise ContractError("evaluation panel self-signature is invalid")
    return result


def rows_for_rank(panel: Mapping[str, Any], rank: int, world_size: int) -> list[dict[str, Any]]:
    """Select panel rows by explicit owner, with a deterministic contiguous fallback."""

    rows = list(panel["rows"])
    owned = sorted(
        [row for row in rows if row.get("worker_rank") == rank],
        key=lambda row: (int(row.get("pipeline_stage", 0)), int(row.get("env_slot", 0))),
    )
    if owned:
        expected = len(rows) // world_size
        if len(rows) % world_size or len(owned) != expected:
            raise ContractError(
                f"panel owner rank {rank} has {len(owned)} rows; expected {expected}"
            )
        return owned
    if world_size != CONTRACT.world_size:
        raise ContractError(
            "panel rows have no worker_rank ownership; only eight-rank fixed-panel "
            "execution can derive the recorded partition"
        )
    expected = len(rows) // world_size
    return rows[rank * expected : (rank + 1) * expected]


def write_json(path: Path | None, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    if path is None:
        print(rendered, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)

#!/usr/bin/env python3
"""Evaluate one OpenPI checkpoint on the matched 500-row MetaWorld MT50 panel."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from path_opd.release_contract import (  # noqa: E402
    ReleaseContractError,
    canonical_json_bytes,
    formal_panel_contract,
    validate_rlinf_host_support,
    verify_panel_digest,
)

try:
    from .common import (
        EVALUATOR_PROVENANCE,
        PROTOCOL,
        RuntimePaths,
        activate_rlinf,
        atomic_write_json,
        checkpoint_weight_file,
        dry_run_plan,
        file_sha256,
        print_json,
        validate_evaluate_paths,
        validate_evaluation_checkpoint,
        validate_task_config,
    )
except ImportError:  # direct ``python benchmarks/.../evaluate.py`` invocation
    from common import (  # type: ignore[no-redef]
        EVALUATOR_PROVENANCE,
        PROTOCOL,
        RuntimePaths,
        activate_rlinf,
        atomic_write_json,
        checkpoint_weight_file,
        dry_run_plan,
        file_sha256,
        print_json,
        validate_evaluate_paths,
        validate_evaluation_checkpoint,
        validate_task_config,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("path_opd", "endpoint_dagger"),
        required=True,
        help="label recorded in every result row",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        choices=(0, 1),
        required=True,
        help="paper training seed represented by the checkpoint",
    )
    parser.add_argument("--evaluation-seed", type=int, default=195)
    parser.add_argument(
        "--panel-mode",
        choices=("formal", "custom"),
        default="custom",
        help="MetaWorld has no published formal panel digest; custom is the honest default",
    )
    parser.add_argument(
        "--panel-sha256",
        type=_sha256_argument,
        help="digest of the generated panel identity (required for a real run)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate paths and print the 500-row plan without importing runtime deps",
    )
    return parser


def row_seed(base_seed: int, task_index: int, trial_id: int) -> int:
    """Seed one matched reset row independently of checkpoint or method."""
    return base_seed + task_index * PROTOCOL.variants_per_task + trial_id


def inference_seed(
    base_seed: int,
    task_index: int,
    trial_id: int,
    replanning_call: int,
) -> int:
    """Seed OpenPI replanning without coupling different evaluation rows."""
    return base_seed * 10_000_000 + task_index * 100_000 + trial_id * 1_000 + replanning_call


def evaluation_protocol() -> dict[str, Any]:
    return {
        "suite": PROTOCOL.suite,
        "config_name": PROTOCOL.config_name,
        "tasks": PROTOCOL.task_count,
        "matched_variants_per_task": PROTOCOL.variants_per_task,
        "rows": PROTOCOL.evaluation_rows,
        "episode_limit_primitive_steps": PROTOCOL.episode_limit,
        "model_action_horizon": PROTOCOL.model_horizon,
        "execution_prefix": PROTOCOL.execution_prefix,
        "physical_action_dims": PROTOCOL.physical_action_dims,
        "ode_steps_k": PROTOCOL.student_steps,
        "metric": "success_once",
        "reset_settle_steps": PROTOCOL.reset_settle_steps,
        "camera_id": PROTOCOL.camera_id,
    }


def _sha256_argument(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise argparse.ArgumentTypeError("expected 64 hexadecimal SHA-256 digits")
    return normalized


def panel_identity_payload(
    task_items: list[tuple[str, str]],
    *,
    task_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the path-free identity that defines the custom MT50 panel."""

    return {
        "schema": "path-opd-metaworld-custom-panel-v1",
        "source": {
            "kind": "rlinf_task_config",
            "task_config_sha256": task_config_sha256,
            "evaluator_implementation": EVALUATOR_PROVENANCE.implementation,
            "evaluator_provenance": EVALUATOR_PROVENANCE.as_dict(),
        },
        "protocol": evaluation_protocol(),
        "tasks": [[name, prompt] for name, prompt in task_items],
    }


def generated_panel_sha256(
    task_items: list[tuple[str, str]],
    *,
    task_config_sha256: str | None = None,
) -> str:
    """Hash the ordered task/prompt panel and its source provenance."""

    payload = panel_identity_payload(task_items, task_config_sha256=task_config_sha256)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _panel_verification(
    args: argparse.Namespace,
    task_items: list[tuple[str, str]],
    *,
    require_supplied: bool,
    task_config_sha256: str | None = None,
) -> dict[str, Any]:
    actual = generated_panel_sha256(task_items, task_config_sha256=task_config_sha256)
    contract = formal_panel_contract("metaworld_mt50")

    def enrich(result: dict[str, Any]) -> dict[str, Any]:
        result.update(
            {
                "identity_schema": "path-opd-metaworld-custom-panel-v1",
                "task_config_sha256": task_config_sha256,
                "provenance": panel_identity_payload(
                    task_items,
                    task_config_sha256=task_config_sha256,
                )["source"],
            }
        )
        return result

    if args.panel_mode == "formal":
        try:
            return enrich(
                verify_panel_digest(
                    "metaworld_mt50",
                    actual,
                    mode="formal",
                    supplied_sha256=args.panel_sha256,
                    rows=PROTOCOL.evaluation_rows,
                ).as_dict()
            )
        except ReleaseContractError as error:
            raise ValueError(str(error)) from error
    if args.panel_sha256 is None:
        if require_supplied:
            raise ValueError(
                "real MetaWorld evaluation requires --panel-sha256 for the custom panel"
            )
        return enrich(
            {
                "benchmark": "metaworld_mt50",
                "mode": "custom",
                "formal": False,
                "custom": True,
                "actual_sha256": actual,
                "expected_sha256": contract.expected_sha256,
                "rows": PROTOCOL.evaluation_rows,
                "protocol": "matched_mt50",
                "verified": False,
                "verification_note": (
                    "dry-run only; pass actual_sha256 as --panel-sha256 for execution"
                ),
            }
        )
    try:
        return enrich(
            verify_panel_digest(
                "metaworld_mt50",
                actual,
                mode="custom",
                supplied_sha256=args.panel_sha256,
                rows=PROTOCOL.evaluation_rows,
            ).as_dict()
        )
    except ReleaseContractError as error:
        raise ValueError(str(error)) from error


def _validate_args(args: argparse.Namespace) -> RuntimePaths:
    if args.evaluation_seed < 0:
        raise ValueError("evaluation-seed must be non-negative")
    if not args.device:
        raise ValueError("device must be non-empty")
    paths = validate_evaluate_paths(
        rlinf_checkout=args.rlinf_checkout,
        checkpoint=args.checkpoint,
        norm_stats=args.norm,
        output=args.output,
        require_existing=not args.dry_run,
    )
    if args.dry_run:
        # A dry-run remains declaration-only, but if a complete checkpoint is
        # present we opportunistically audit its sidecar and expose failures
        # in the plan instead of pretending the contract was verified.
        try:
            args.checkpoint_contract = validate_evaluation_checkpoint(
                paths,
                method=args.method,
                training_seed=args.training_seed,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            args.checkpoint_contract = {
                "verified": False,
                "verification_error": str(error),
                "verification_note": "dry-run only; no checkpoint sidecar was accepted",
            }
    else:
        args.checkpoint_contract = validate_evaluation_checkpoint(
            paths,
            method=args.method,
            training_seed=args.training_seed,
        )
    return paths


def _seed_all(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_config(checkpoint: Path, norm: Path) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "model_path": str(checkpoint),
            "openpi_data": {"norm_stats_path": str(norm)},
            "openpi": {
                "config_name": PROTOCOL.config_name,
                "num_images_in_input": 1,
                "noise_method": "flow_sde",
                "joint_logprob": False,
                "action_horizon": PROTOCOL.model_horizon,
                "action_chunk": PROTOCOL.execution_prefix,
                "action_env_dim": PROTOCOL.physical_action_dims,
                "num_steps": PROTOCOL.student_steps,
                "train_expert_only": False,
                "add_value_head": False,
                "is_nft": False,
            },
        }
    )


def _load_policy(paths: RuntimePaths, device: Any) -> Any:
    from rlinf.models.embodiment.openpi import get_model

    if paths.checkpoint is None:
        raise ValueError("checkpoint is required")
    model = get_model(_model_config(paths.checkpoint, paths.norm_stats))
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    return model


def _find_task_selector(env: Any) -> Any:
    from metaworld.wrappers import RandomTaskSelectWrapper

    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, RandomTaskSelectWrapper):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("MetaWorld RandomTaskSelectWrapper was not found")


def _set_camera(env: Any) -> None:
    model = getattr(getattr(env, "unwrapped", env), "model", None)
    if model is None:
        current = env
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            model = getattr(current, "model", None)
            if model is not None:
                break
            current = getattr(current, "env", None)
    if model is None:
        raise RuntimeError("MetaWorld model is not reachable through gym wrappers")
    model.cam_pos[PROTOCOL.camera_id] = [0.75, 0.075, 0.7]


def _existing_rows(
    path: Path,
    *,
    checkpoint_sha256: str,
    normalizer_sha256: str,
    method: str,
    training_seed: int,
    evaluation_seed_value: int,
    panel_mode: str = "custom",
    panel_sha256: str | None = None,
    panel_task_config_sha256: str | None = None,
    panel_identity_schema: str | None = None,
    panel_provenance: dict[str, Any] | None = None,
    checkpoint_release_contract_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], set[tuple[int, int]]]:
    rows: list[dict[str, Any]] = []
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    if not path.exists():
        return rows, set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL row {line_number}: {error}") from error
        if row.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("rows.jsonl belongs to a different checkpoint")
        if row.get("normalizer_sha256") != normalizer_sha256:
            raise ValueError("rows.jsonl belongs to different normalization statistics")
        expected_identity = (method, training_seed, evaluation_seed_value)
        actual_identity = (
            row.get("method"),
            row.get("training_seed"),
            row.get("evaluation_seed"),
        )
        if actual_identity != expected_identity:
            raise ValueError("rows.jsonl belongs to a different evaluation run")
        if row.get("evaluator_implementation") != EVALUATOR_PROVENANCE.implementation:
            raise ValueError("rows.jsonl has incompatible evaluator provenance")
        if row.get("panel_mode", panel_mode) != panel_mode:
            raise ValueError("rows.jsonl uses a different panel mode")
        if panel_sha256 is not None and row.get("panel_sha256") != panel_sha256:
            raise ValueError("rows.jsonl belongs to a different panel identity")
        if (
            panel_task_config_sha256 is not None
            and row.get("panel_task_config_sha256") != panel_task_config_sha256
        ):
            raise ValueError("rows.jsonl belongs to a different panel task-config provenance")
        if (
            panel_identity_schema is not None
            and row.get("panel_identity_schema") != panel_identity_schema
        ):
            raise ValueError("rows.jsonl uses an incompatible panel identity schema")
        if panel_provenance is not None and row.get("panel_provenance") != panel_provenance:
            raise ValueError("rows.jsonl has incompatible panel provenance")
        if (
            checkpoint_release_contract_sha256 is not None
            and row.get("checkpoint_release_contract_sha256")
            != checkpoint_release_contract_sha256
        ):
            raise ValueError("rows.jsonl belongs to a different checkpoint release contract")
        key = (row.get("task_index"), row.get("trial_id"))
        if not all(isinstance(value, int) for value in key):
            raise ValueError(f"invalid task key in JSONL row {line_number}")
        if not 0 <= key[0] < PROTOCOL.task_count:
            raise ValueError(f"out-of-range task index in JSONL row {line_number}")
        if not 0 <= key[1] < PROTOCOL.variants_per_task:
            raise ValueError(f"out-of-range trial id in JSONL row {line_number}")
        if row.get("protocol") != evaluation_protocol():
            raise ValueError("rows.jsonl uses a different evaluation protocol")
        rows.append(row)
        latest[key] = row
    completed = {key for key, row in latest.items() if row.get("exception") is None}
    return rows, completed


def _task_variant_sha256(task: Any) -> str:
    task_data = getattr(task, "data", b"")
    if not isinstance(task_data, bytes):
        task_data = bytes(task_data)
    return hashlib.sha256(task_data).hexdigest()


def _validate_completed_task_variant(
    row: Mapping[str, Any], *, task_index: int, trial_id: int, actual_sha256: str
) -> None:
    expected_sha256 = row.get("task_variant_sha256")
    if expected_sha256 != actual_sha256:
        raise ValueError(
            "completed MetaWorld row task variant mismatch "
            f"for task_index={task_index}, trial_id={trial_id}"
        )


def _validate_all_completed_task_variants(
    gym: Any,
    task_items: list[tuple[str, str]],
    *,
    evaluation_seed_value: int,
    completed_rows: Mapping[tuple[int, int], Mapping[str, Any]],
) -> None:
    for task_index, (env_name, _prompt) in enumerate(task_items):
        env = gym.make(
            "Meta-World/MT1",
            env_name=env_name,
            seed=evaluation_seed_value,
            render_mode="rgb_array",
            camera_id=PROTOCOL.camera_id,
            disable_env_checker=True,
        )
        try:
            selector = _find_task_selector(env)
            selector.toggle_sample_tasks_on_reset(False)
            if len(selector.tasks) < PROTOCOL.variants_per_task:
                raise RuntimeError(
                    f"{env_name} exposes {len(selector.tasks)} variants; "
                    f"expected at least {PROTOCOL.variants_per_task}"
                )
            for trial_id in range(PROTOCOL.variants_per_task):
                key = (task_index, trial_id)
                row = completed_rows.get(key)
                if row is None:
                    raise ValueError(
                        "complete MetaWorld evaluation is missing a completed task row "
                        f"for task_index={task_index}, trial_id={trial_id}"
                    )
                _validate_completed_task_variant(
                    row,
                    task_index=task_index,
                    trial_id=trial_id,
                    actual_sha256=_task_variant_sha256(selector.tasks[trial_id]),
                )
        finally:
            env.close()


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    checkpoint_sha256: str,
    normalizer_sha256: str,
    method: str,
    training_seed: int,
    evaluation_seed_value: int,
    rows_filename: str = "rows.jsonl",
    panel_mode: str = "custom",
    panel_sha256: str | None = None,
    panel_task_config_sha256: str | None = None,
    panel_identity_schema: str | None = None,
    panel_provenance: dict[str, Any] | None = None,
    checkpoint_release_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Summarize the latest attempt for every unique task/variant key."""
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["task_index"]), int(row["trial_id"]))
        latest[key] = row
    successful_rows = [row for row in latest.values() if row.get("exception") is None]
    failures = [row for row in latest.values() if row.get("exception") is not None]
    successes = sum(bool(row.get("success_once")) for row in successful_rows)
    per_task = []
    for task_index in range(PROTOCOL.task_count):
        task_rows = [
            row
            for (index, _), row in latest.items()
            if index == task_index and row.get("exception") is None
        ]
        task_successes = sum(bool(row.get("success_once")) for row in task_rows)
        per_task.append(
            {
                "task_index": task_index,
                "task": task_rows[0].get("task") if task_rows else None,
                "evaluated": len(task_rows),
                "successes": task_successes,
                "success_once_rate": (task_successes / len(task_rows) if task_rows else None),
            }
        )
    if failures:
        status = "FAILED"
    elif len(successful_rows) == PROTOCOL.evaluation_rows:
        status = "COMPLETE"
    else:
        status = "IN_PROGRESS"
    return {
        "schema_version": 1,
        "benchmark": PROTOCOL.suite,
        "status": status,
        "evaluator_implementation": EVALUATOR_PROVENANCE.implementation,
        "evaluator_provenance": EVALUATOR_PROVENANCE.as_dict(),
        "protocol": evaluation_protocol(),
        "method": method,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed_value,
        # The historical evaluator and panel are not released byte-for-byte.
        # A complete row set is therefore still a reconstructed qualification.
        "formal_benchmark": False,
        "non_formal": True,
        "evaluator_equivalence": False,
        "qualification": True,
        "checkpoint_sha256": checkpoint_sha256,
        "normalizer_sha256": normalizer_sha256,
        "panel_mode": panel_mode,
        "panel_sha256": panel_sha256,
        "panel_task_config_sha256": panel_task_config_sha256,
        "panel_identity_schema": panel_identity_schema,
        "panel_provenance": panel_provenance,
        "checkpoint_release_contract_sha256": checkpoint_release_contract_sha256,
        "rows_file": rows_filename,
        "records_in_jsonl": len(rows),
        "unique_rows": len(latest),
        "evaluated_rows": len(successful_rows),
        "exception_rows": len(failures),
        "expected_rows": PROTOCOL.evaluation_rows,
        "successes": successes,
        "success_once_rate": (successes / len(successful_rows) if successful_rows else None),
        "per_task": per_task,
        "private_paths_omitted": True,
    }


def _write_summary(
    output: Path,
    rows: list[dict[str, Any]],
    *,
    checkpoint_sha256: str,
    normalizer_sha256: str,
    args: argparse.Namespace,
    panel_sha256: str | None = None,
    panel_task_config_sha256: str | None = None,
    panel_identity_schema: str | None = None,
    panel_provenance: dict[str, Any] | None = None,
    checkpoint_release_contract_sha256: str | None = None,
    checkpoint_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summarize_rows(
        rows,
        checkpoint_sha256=checkpoint_sha256,
        normalizer_sha256=normalizer_sha256,
        method=args.method,
        training_seed=args.training_seed,
        evaluation_seed_value=args.evaluation_seed,
        panel_mode=args.panel_mode,
        panel_sha256=panel_sha256,
        panel_task_config_sha256=panel_task_config_sha256,
        panel_identity_schema=panel_identity_schema,
        panel_provenance=panel_provenance,
        checkpoint_release_contract_sha256=checkpoint_release_contract_sha256,
    )
    if checkpoint_contract is not None:
        summary["checkpoint_contract"] = checkpoint_contract
    rows_path = output / "rows.jsonl"
    if rows_path.is_file():
        summary["rows_file_sha256"] = file_sha256(rows_path)
    atomic_write_json(output / "summary.json", summary)
    return summary


def _run(args: argparse.Namespace, paths: RuntimePaths) -> int:
    # Validate the JSON sidecar before importing simulator/CUDA dependencies.
    # This is also a guard for callers that invoke _run directly instead of
    # going through main().
    checkpoint_contract = getattr(args, "checkpoint_contract", None)
    if (
        not isinstance(checkpoint_contract, dict)
        or not checkpoint_contract.get("verified")
        or checkpoint_contract.get("method") != args.method
        or checkpoint_contract.get("training_seed") != args.training_seed
    ):
        checkpoint_contract = validate_evaluation_checkpoint(
            paths,
            method=args.method,
            training_seed=args.training_seed,
        )
    validate_rlinf_host_support(
        paths.rlinf_checkout,
        benchmark="metaworld_mt50",
        patch_path=Path(__file__).resolve().parents[2]
        / "integrations"
        / "rlinf"
        / "path-opd-host-support.patch",
        require_openpi_distribution=True,
    )
    import gymnasium as gym
    import metaworld
    import numpy as np
    import torch

    activate_rlinf(paths.rlinf_checkout)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable; use --dry-run to audit")
    device = torch.device(args.device)
    if paths.checkpoint is None:
        raise ValueError("checkpoint is required")
    checkpoint_file = checkpoint_weight_file(paths.checkpoint)
    checkpoint_sha256 = file_sha256(checkpoint_file)
    normalizer_sha256 = file_sha256(paths.norm_stats)
    task_items = validate_task_config(paths.task_config)
    task_config_sha256 = file_sha256(paths.task_config)
    panel_verification = _panel_verification(
        args,
        task_items,
        require_supplied=True,
        task_config_sha256=task_config_sha256,
    )
    panel_sha256 = str(panel_verification["actual_sha256"])
    panel_task_config_sha256 = panel_verification.get("task_config_sha256")
    panel_identity_schema = str(panel_verification["identity_schema"])
    panel_provenance = panel_verification.get("provenance")
    checkpoint_release_contract_sha256 = str(checkpoint_contract["release_contract_sha256"])
    paths.output.mkdir(parents=True, exist_ok=True)
    rows_path = paths.output / "rows.jsonl"
    rows, completed = _existing_rows(
        rows_path,
        checkpoint_sha256=checkpoint_sha256,
        normalizer_sha256=normalizer_sha256,
        method=args.method,
        training_seed=args.training_seed,
        evaluation_seed_value=args.evaluation_seed,
        panel_mode=args.panel_mode,
        panel_sha256=panel_sha256,
        panel_task_config_sha256=panel_task_config_sha256,
        panel_identity_schema=panel_identity_schema,
        panel_provenance=panel_provenance,
        checkpoint_release_contract_sha256=checkpoint_release_contract_sha256,
    )
    completed_rows = {
        (int(row["task_index"]), int(row["trial_id"])): row
        for row in rows
        if (int(row["task_index"]), int(row["trial_id"])) in completed
    }
    _seed_all(args.evaluation_seed)
    metaworld.register_mw_envs()
    if len(completed) == PROTOCOL.evaluation_rows:
        _validate_all_completed_task_variants(
            gym,
            task_items,
            evaluation_seed_value=args.evaluation_seed,
            completed_rows=completed_rows,
        )
        summary = _write_summary(
            paths.output,
            rows,
            checkpoint_sha256=checkpoint_sha256,
            normalizer_sha256=normalizer_sha256,
            args=args,
            panel_sha256=panel_sha256,
            panel_task_config_sha256=panel_task_config_sha256,
            panel_identity_schema=panel_identity_schema,
            panel_provenance=panel_provenance,
            checkpoint_release_contract_sha256=checkpoint_release_contract_sha256,
            checkpoint_contract=checkpoint_contract,
        )
        print_json(summary)
        return 0

    policy = _load_policy(paths, device)
    started_at = time.time()
    with rows_path.open("a", encoding="utf-8", buffering=1) as stream:
        for task_index, (env_name, prompt) in enumerate(task_items):
            env = gym.make(
                "Meta-World/MT1",
                env_name=env_name,
                seed=args.evaluation_seed,
                render_mode="rgb_array",
                camera_id=PROTOCOL.camera_id,
                disable_env_checker=True,
            )
            try:
                _set_camera(env)
                selector = _find_task_selector(env)
                selector.toggle_sample_tasks_on_reset(False)
                if len(selector.tasks) < PROTOCOL.variants_per_task:
                    raise RuntimeError(
                        f"{env_name} exposes {len(selector.tasks)} variants; "
                        f"expected at least {PROTOCOL.variants_per_task}"
                    )
                for trial_id in range(PROTOCOL.variants_per_task):
                    key = (task_index, trial_id)
                    task = selector.tasks[trial_id]
                    task_variant_sha256 = _task_variant_sha256(task)
                    if key in completed:
                        _validate_completed_task_variant(
                            completed_rows[key],
                            task_index=task_index,
                            trial_id=trial_id,
                            actual_sha256=task_variant_sha256,
                        )
                        continue
                    reset_seed = row_seed(args.evaluation_seed, task_index, trial_id)
                    random.seed(reset_seed)
                    np.random.seed(reset_seed)
                    env.unwrapped.set_task(task)
                    observation, _ = env.reset(seed=reset_seed)
                    for _ in range(PROTOCOL.reset_settle_steps):
                        observation, _, _, _, _ = env.step(
                            np.zeros(PROTOCOL.physical_action_dims, dtype=np.float32)
                        )
                    initial_observation_sha256 = hashlib.sha256(
                        np.asarray(observation).tobytes()
                    ).hexdigest()
                    action_plan: collections.deque[np.ndarray] = collections.deque()
                    success = False
                    steps_executed = 0
                    replanning_calls = 0
                    first_action: list[float] | None = None
                    exception: str | None = None
                    try:
                        for step in range(PROTOCOL.episode_limit):
                            if not action_plan:
                                sample_seed = inference_seed(
                                    args.evaluation_seed,
                                    task_index,
                                    trial_id,
                                    replanning_calls,
                                )
                                _seed_all(sample_seed)
                                image = np.asarray(env.render())[::-1, ::-1]
                                env_obs = {
                                    "main_images": torch.from_numpy(np.ascontiguousarray(image))[
                                        None
                                    ].to(device),
                                    "states": torch.from_numpy(
                                        np.asarray(observation[:4], dtype=np.float32)
                                    )[None].to(device),
                                    "task_descriptions": [prompt],
                                    "wrist_images": None,
                                    "extra_view_images": None,
                                }
                                with torch.no_grad():
                                    actions, _ = policy.predict_action_batch(
                                        env_obs=env_obs,
                                        mode="eval",
                                        compute_values=False,
                                    )
                                actions_array = actions.detach().float().cpu().numpy()[0]
                                expected_shape = (
                                    PROTOCOL.model_horizon,
                                    PROTOCOL.physical_action_dims,
                                )
                                if actions_array.shape != expected_shape:
                                    raise RuntimeError(
                                        f"{env_name}: action shape {actions_array.shape} "
                                        f"!= {expected_shape}"
                                    )
                                executed_chunk = actions_array[: PROTOCOL.execution_prefix]
                                if not np.isfinite(executed_chunk).all():
                                    raise RuntimeError(f"{env_name}: non-finite action")
                                if first_action is None:
                                    first_action = executed_chunk[0].tolist()
                                action_plan.extend(executed_chunk)
                                replanning_calls += 1
                            observation, _, terminated, truncated, info = env.step(
                                action_plan.popleft().astype(np.float32, copy=False)
                            )
                            steps_executed = step + 1
                            if info.get("success", 0):
                                success = True
                                break
                            if terminated or truncated:
                                break
                    except Exception as error:  # preserve a row before failing closed
                        exception = f"{type(error).__name__}: {error}"
                    row = {
                        "schema_version": 1,
                        "benchmark": PROTOCOL.suite,
                        "evaluator_implementation": EVALUATOR_PROVENANCE.implementation,
                        "evaluator_provenance": EVALUATOR_PROVENANCE.as_dict(),
                        "checkpoint_sha256": checkpoint_sha256,
                        "normalizer_sha256": normalizer_sha256,
                        "checkpoint_release_contract_sha256": checkpoint_release_contract_sha256,
                        "panel_mode": args.panel_mode,
                        "panel_sha256": panel_sha256,
                        "panel_task_config_sha256": panel_task_config_sha256,
                        "panel_identity_schema": panel_identity_schema,
                        "panel_provenance": panel_provenance,
                        "panel_formal": bool(panel_verification["formal"]),
                        "method": args.method,
                        "training_seed": args.training_seed,
                        "evaluation_seed": args.evaluation_seed,
                        "task_index": task_index,
                        "task": env_name,
                        "trial_id": trial_id,
                        "reset_seed": reset_seed,
                        "task_variant_sha256": task_variant_sha256,
                        "initial_observation_sha256": initial_observation_sha256,
                        "success_once": success,
                        "steps_executed": steps_executed,
                        "replanning_calls": replanning_calls,
                        "first_action": first_action,
                        "exception": exception,
                        "protocol": evaluation_protocol(),
                        "elapsed_seconds": time.time() - started_at,
                        "private_paths_omitted": True,
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    rows.append(row)
                    _write_summary(
                        paths.output,
                        rows,
                        checkpoint_sha256=checkpoint_sha256,
                        normalizer_sha256=normalizer_sha256,
                        args=args,
                        panel_sha256=panel_sha256,
                        panel_task_config_sha256=panel_task_config_sha256,
                        panel_identity_schema=panel_identity_schema,
                        panel_provenance=panel_provenance,
                        checkpoint_release_contract_sha256=checkpoint_release_contract_sha256,
                        checkpoint_contract=checkpoint_contract,
                    )
                    print(json.dumps(row, sort_keys=True), flush=True)
                    if exception is not None:
                        raise RuntimeError(exception)
                    completed.add(key)
            finally:
                env.close()
    summary = _write_summary(
        paths.output,
        rows,
        checkpoint_sha256=checkpoint_sha256,
        normalizer_sha256=normalizer_sha256,
        args=args,
        panel_sha256=panel_sha256,
        panel_task_config_sha256=panel_task_config_sha256,
        panel_identity_schema=panel_identity_schema,
        panel_provenance=panel_provenance,
        checkpoint_release_contract_sha256=checkpoint_release_contract_sha256,
        checkpoint_contract=checkpoint_contract,
    )
    if summary["status"] != "COMPLETE":
        raise RuntimeError(
            f"evaluation ended with {summary['evaluated_rows']}/{PROTOCOL.evaluation_rows} rows"
        )
    print_json(summary)
    return 0


def _synthetic_smoke(argv: list[str]) -> int:
    """Run the dependency-light train/checkpoint/panel/trace evaluator proof."""

    parser = argparse.ArgumentParser(
        prog="metaworld-evaluate --synthetic-smoke",
        description=(
            "Run the synthetic MetaWorld evaluator contract; no RLinf checkout, "
            "model, panel, simulator, or CUDA device is used."
        ),
    )
    parser.add_argument("--synthetic-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("artifacts/evaluator-smoke/metaworld_mt50")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    from path_opd.adapters.benchmark_smoke import (
        SyntheticSmokeConfig,
        run_evaluator_synthetic_smoke,
        write_report,
    )

    report = run_evaluator_synthetic_smoke(
        "metaworld_mt50",
        args.work_dir,
        entrypoint="benchmarks/metaworld_mt50/evaluate.py",
        config=SyntheticSmokeConfig(
            updates=args.updates,
            batch_size=args.batch_size,
            seed=args.seed,
        ),
    )
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        write_report(args.output, report)
        print(args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--synthetic-smoke" in raw_argv:
        try:
            return _synthetic_smoke(raw_argv)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            print(f"metaworld-evaluate synthetic smoke: {error}", file=sys.stderr)
            return 2
    args = build_parser().parse_args(raw_argv)
    try:
        paths = _validate_args(args)
        if args.dry_run:
            plan = dry_run_plan(
                "evaluate",
                paths,
                seed=args.evaluation_seed,
            )
            task_items = (
                validate_task_config(paths.task_config)
                if paths.task_config.is_file()
                else [
                    (f"task-{index}-v3", f"synthetic task {index}")
                    for index in range(PROTOCOL.task_count)
                ]
            )
            task_config_sha256 = (
                file_sha256(paths.task_config) if paths.task_config.is_file() else None
            )
            plan["evaluation"].update(
                {
                    "method": args.method,
                    "training_seed": args.training_seed,
                    "protocol": evaluation_protocol(),
                    "checkpoint_contract": args.checkpoint_contract,
                    "panel": _panel_verification(
                        args,
                        task_items,
                        require_supplied=False,
                        task_config_sha256=task_config_sha256,
                    ),
                }
            )
            print_json(plan)
            return 0
        return _run(args, paths)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"metaworld-evaluate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

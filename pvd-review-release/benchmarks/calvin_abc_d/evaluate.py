#!/usr/bin/env python3
"""Evaluate a CALVIN ABC-D checkpoint on the fixed Official-D panel.

The evaluator is deliberately a small runtime seam.  It keeps the official
panel, model/checkpoint, normalizer, task metadata, and simulator checkout
caller supplied; importing this module or running ``--dry-run`` does not load
CUDA, CALVIN, RLinf, Hydra, or OpenPI.  A real run is inference-only and
records row coverage, fixed-noise evidence, parameter immutability, and the
Official-D SR1--SR5 metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
for _path in (_REPOSITORY_ROOT, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from path_opd.release_contract import (  # noqa: E402
    ReleaseContractError,
    git_revision,
    tree_sha256,
    validate_rlinf_host_support,
)
from path_opd.release_contract import (  # noqa: E402
    validate_checkpoint_manifest as validate_release_checkpoint_manifest,
)

try:
    from .common import (
        PROTOCOL,
        ContractError,
        RuntimePaths,
        activate_rlinf,
        atomic_write_json,
        checkpoint_artifact,
        dry_run_plan,
        load_checkpoint_manifest,
        paper_contract,
        rows_for_rank,
        sha256_file,
        validate_checkpoint_manifest,
        validate_evaluate_paths,
        validate_panel,
    )
    from .environment import CalvinEnvironmentAdapter, policy_observation, prepare_actions
except ImportError:  # direct ``python benchmarks/calvin_abc_d/evaluate.py`` invocation
    from benchmarks.calvin_abc_d.common import (  # type: ignore[no-redef]
        PROTOCOL,
        ContractError,
        RuntimePaths,
        activate_rlinf,
        atomic_write_json,
        checkpoint_artifact,
        dry_run_plan,
        load_checkpoint_manifest,
        paper_contract,
        rows_for_rank,
        sha256_file,
        validate_checkpoint_manifest,
        validate_evaluate_paths,
        validate_panel,
    )
    from benchmarks.calvin_abc_d.environment import (  # type: ignore[no-redef]
        CalvinEnvironmentAdapter,
        policy_observation,
        prepare_actions,
    )


SCHEMA = "path-opd-calvin-evaluation-v1"
ROW_SCHEMA = "path-opd-calvin-sequence-row-v1"
NOISE_LITERAL = "path-opd-calvin-official-d-noise-v1"

RESULT_SCHEMA = {
    "schema": SCHEMA,
    "status": "COMPLETE",
    "benchmark": PROTOCOL.benchmark,
    "formal_benchmark": "boolean",
    "non_formal": "boolean",
    "qualification": "boolean",
    "protocol": {
        "name": "official_d",
        "rows": PROTOCOL.evaluation_rows,
        "sequence_length": PROTOCOL.sequence_length,
        "subtask_limit_primitive_steps": PROTOCOL.evaluation_subtask_limit,
        "solver_steps": PROTOCOL.student_steps,
        "model_horizon": PROTOCOL.model_horizon,
        "execution_prefix": PROTOCOL.execution_prefix,
        "physical_action_dims": PROTOCOL.physical_action_dims,
    },
    "coverage": {
        "expected_rows": "integer",
        "actual_rows": "integer",
        "unique_identity_sha256": "integer",
        "missing_rows": "integer",
        "duplicate_rows": "integer",
        "exception_rows": "integer",
    },
    "metrics": {
        "average_completed_sequence_length": "float",
        "SR1": "float",
        "SR2": "float",
        "SR3": "float",
        "SR4": "float",
        "SR5": "float",
    },
    "rows": [
        {
            "sequence_index": "integer",
            "identity_sha256": "sha256",
            "completed_prefix_length": "integer",
            "primitive_steps": "integer",
            "policy_calls": "integer",
            "exception": "null|string",
        }
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-checkout", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--normalization-stats",
        "--norm",
        dest="normalization_stats",
        type=Path,
        required=True,
    )
    parser.add_argument("--environment-assets", type=Path, required=True)
    parser.add_argument("--task-oracle-annotations", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("path_opd", "endpoint_dagger"), required=True)
    parser.add_argument("--training-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--evaluation-seed", type=int, default=20260819)
    parser.add_argument("--world-size", type=int, default=PROTOCOL.world_size)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--horizon",
        type=int,
        default=PROTOCOL.evaluation_subtask_limit,
        help="maximum primitive steps per subtask; the formal value is 360",
    )
    parser.add_argument(
        "--allow-custom-panel",
        action="store_true",
        help=(
            "permit a structurally valid qualification panel with 1..1,000 rows; "
            "the eight-rank runner requires a row count divisible by eight"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--contract-output",
        type=Path,
        help="write the dry-run contract here instead of standard output",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> RuntimePaths:
    if args.world_size != PROTOCOL.world_size:
        raise ContractError(f"Official-D evaluation requires world_size={PROTOCOL.world_size}")
    if args.evaluation_seed < 0:
        raise ContractError("evaluation-seed must be non-negative")
    if args.horizon < 1 or args.horizon > PROTOCOL.evaluation_subtask_limit:
        raise ContractError(f"horizon must be between one and {PROTOCOL.evaluation_subtask_limit}")
    if not args.device:
        raise ContractError("device must be non-empty")
    paths = validate_evaluate_paths(
        rlinf_checkout=args.rlinf_checkout,
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        normalization_stats=args.normalization_stats,
        environment_assets=args.environment_assets,
        task_oracle_annotations=args.task_oracle_annotations,
        official_d_panel=args.panel,
        output=args.output,
        require_existing=not args.dry_run,
        allow_custom_panel=args.allow_custom_panel,
    )
    if not args.dry_run:
        if paths.checkpoint is None:
            raise ContractError("checkpoint is required for evaluation")
        # A raw model file is not enough to establish the paper protocol.  The
        # training-side manifest binds method, seed, solver dimensions, and
        # normalizer bytes, so evaluation fails closed when it is absent.
        checkpoint_artifact(paths.checkpoint)
        manifest = load_checkpoint_manifest(paths.checkpoint)
        checkpoint_digest = sha256_file(checkpoint_artifact(paths.checkpoint))
        recorded_checkpoint_digest = manifest.get("checkpoint_sha256")
        if (
            recorded_checkpoint_digest is not None
            and recorded_checkpoint_digest != checkpoint_digest
        ):
            raise ContractError("checkpoint file digest does not match manifest")
        validate_checkpoint_manifest(
            manifest,
            method=args.method,
            seed=args.training_seed,
            world_size=args.world_size,
            normalizer_sha256=sha256_file(paths.normalization_stats),
        )
        try:
            release_manifest = manifest.get("release_contract")
            if not isinstance(release_manifest, Mapping):
                raise ContractError("checkpoint lacks release contract manifest")
            assets = {
                name: tree_sha256(path)
                for name, path in {
                    "base_model": paths.base_model,
                    "normalization_stats": paths.normalization_stats,
                    "environment_assets": paths.environment_assets,
                    "task_oracle_annotations": paths.task_oracle_annotations,
                }.items()
                if path is not None
            }
            code = {
                "release_source_sha256": tree_sha256(
                    _REPOSITORY_ROOT, include=("src", "benchmarks", "configs")
                ),
                "rlinf_revision": git_revision(paths.rlinf_checkout),
            }
            validate_release_checkpoint_manifest(
                release_manifest,
                expected={
                    "benchmark": PROTOCOL.benchmark,
                    "method": args.method,
                    "seed": args.training_seed,
                    "world_size": args.world_size,
                    "contract": paper_contract(),
                },
                assets=assets,
                code=code,
            )
        except ReleaseContractError as error:
            raise ContractError(f"checkpoint release contract mismatch: {error}") from error
    return paths


def _device_for_rank(torch: Any, requested: str, local_rank: int) -> Any:
    if requested == "cuda":
        return torch.device("cuda", local_rank)
    device = torch.device(requested)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", local_rank)
    return device


def _seed_all(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def noise_seed(identity: str, subtask_index: int, chunk_index: int, evaluation_seed: int) -> int:
    material = (
        NOISE_LITERAL.encode("ascii")
        + b"\0"
        + str(evaluation_seed).encode("ascii")
        + b"\0"
        + identity.encode("ascii")
        + b"\0"
        + str(subtask_index).encode("ascii")
        + b"\0"
        + str(chunk_index).encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def fixed_noise(
    identity: str,
    subtask_index: int,
    chunk_index: int,
    evaluation_seed: int,
    device: Any,
) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(noise_seed(identity, subtask_index, chunk_index, evaluation_seed))
    return torch.randn(
        (1, PROTOCOL.model_horizon, PROTOCOL.model_action_dims),
        generator=generator,
        dtype=torch.float32,
    ).to(device)


def _model_config(model_root: Path, normalizer: Path) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "model_path": str(model_root),
            "openpi_data": {"norm_stats_path": str(normalizer)},
            "openpi": {
                "config_name": PROTOCOL.config_name,
                "num_images_in_input": 2,
                "noise_method": "flow_sde",
                "joint_logprob": False,
                "action_horizon": PROTOCOL.model_horizon,
                "action_chunk": PROTOCOL.execution_prefix,
                "action_env_dim": PROTOCOL.physical_action_dims,
                "num_steps": PROTOCOL.student_steps,
                "train_expert_only": True,
                "add_value_head": False,
                "value_after_vlm": False,
                "is_nft": False,
            },
        }
    )


def _extract_model_state(payload: Any, torch: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value
        if payload and all(
            isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()
        ):
            return payload
    raise RuntimeError("checkpoint does not contain a model state mapping")


def _load_policy(paths: RuntimePaths, device: Any) -> Any:
    import torch

    if paths.rlinf_checkout is None:
        raise ContractError("rlinf-checkout is required for evaluation")
    # Bind the requested checkout before importing any RLinf submodule.  A
    # globally installed package must never silently win over the declared
    # source revision.
    activate_rlinf(paths.rlinf_checkout)
    from rlinf.models.embodiment.openpi import get_model

    model = get_model(_model_config(paths.base_model, paths.normalization_stats)).to(device)
    payload = torch.load(
        checkpoint_artifact(paths.checkpoint), map_location="cpu", weights_only=False
    )
    model.load_state_dict(_extract_model_state(payload, torch), strict=True)
    model.eval().requires_grad_(False)
    return model


def _parameter_sha256(model: Any) -> str:
    import torch

    digest = hashlib.sha256()
    entries = list(model.named_parameters()) + list(model.named_buffers())
    for name, value in sorted(entries):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def action_chunk(
    model: Any,
    observation: Mapping[str, Any],
    language: str,
    identity: str,
    subtask_index: int,
    chunk_index: int,
    evaluation_seed: int,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    policy_input = policy_observation(observation, language)
    policy_input = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in policy_input.items()
    }
    noise = fixed_noise(identity, subtask_index, chunk_index, evaluation_seed, device)
    with torch.inference_mode():
        actions, result = model.predict_action_batch(
            policy_input,
            mode="eval",
            compute_values=False,
            initial_noise=noise,
        )
    chains = result.get("forward_inputs", {}).get("chains")
    expected = (1, PROTOCOL.student_steps + 1, PROTOCOL.model_horizon, PROTOCOL.model_action_dims)
    if chains is None or tuple(chains.shape) != expected:
        actual = None if chains is None else tuple(chains.shape)
        raise RuntimeError(f"CALVIN K8 chain shape drift: {actual} != {expected}")
    if not torch.equal(chains[:, -1], result.get("model_actions")):
        raise RuntimeError("final K8 chain state is not model_actions")
    if tuple(actions.shape) != (1, PROTOCOL.execution_prefix, PROTOCOL.physical_action_dims):
        raise RuntimeError(f"CALVIN physical action shape drift: {tuple(actions.shape)}")
    prepared = prepare_actions(actions)
    try:
        final_chain = chains[:, -1].detach().cpu().contiguous().view(torch.uint8).numpy()
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError("CALVIN model did not return a tensor chain") from error
    return prepared[0], {
        "noise_seed": noise_seed(identity, subtask_index, chunk_index, evaluation_seed),
        "chain_final_sha256": hashlib.sha256(final_chain.tobytes()).hexdigest(),
    }


def _initial_state(row: Mapping[str, Any]) -> tuple[Any, Any]:
    state = row.get("initial_state")
    if isinstance(state, Mapping) and "robot_obs" in state and "scene_obs" in state:
        return state["robot_obs"], state["scene_obs"]
    try:
        from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
    except ImportError as error:  # pragma: no cover - runtime-only dependency
        raise RuntimeError(
            "CALVIN initial-state decoding requires calvin_agent; supply the official runtime"
        ) from error
    return get_env_state_for_initial_condition(state)


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


def evaluate_sequence(
    environment: CalvinEnvironmentAdapter,
    model: Any,
    task_oracle: Any,
    annotations: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    device: Any,
    evaluation_seed: int,
    horizon: int = PROTOCOL.evaluation_subtask_limit,
) -> dict[str, Any]:
    robot_obs, scene_obs = _initial_state(row)
    observation = environment.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    completed = 0
    primitive_steps = 0
    policy_calls = 0
    subtask_rows: list[dict[str, Any]] = []
    noise_evidence: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        subtasks = list(row["subtasks"])
        for subtask_index, subtask in enumerate(subtasks):
            values = annotations[subtask]
            if not isinstance(values, list | tuple) or len(values) != 1:
                raise RuntimeError(f"Official-D annotation count for {subtask!r} is not one")
            language = str(values[0])
            start_info = environment.get_info()
            succeeded = False
            subtask_steps = 0
            subtask_calls = 0
            chunk_index = 0
            while subtask_steps < horizon and not succeeded:
                chunk, evidence = action_chunk(
                    model,
                    observation,
                    language,
                    str(row["identity_sha256"]),
                    subtask_index,
                    chunk_index,
                    evaluation_seed,
                    device,
                )
                if len(noise_evidence) < 2:
                    noise_evidence.append(evidence)
                else:
                    noise_evidence[-1] = evidence
                policy_calls += 1
                subtask_calls += 1
                chunk_index += 1
                for action in chunk:
                    if subtask_steps >= horizon:
                        break
                    observation, reward, done, current_info = environment.step(action)
                    primitive_steps += 1
                    subtask_steps += 1
                    if _scalar(reward) != 0 or bool(_scalar(done)):
                        raise RuntimeError(
                            f"Official-D environment returned reward={reward!r}, done={done!r}"
                        )
                    achieved = task_oracle.get_task_info_for_set(
                        start_info, current_info, {subtask}
                    )
                    if len(achieved) > 0:
                        succeeded = True
                        break
            subtask_rows.append(
                {
                    "subtask_index": subtask_index,
                    "task": subtask,
                    "language": language,
                    "success": succeeded,
                    "primitive_steps": subtask_steps,
                    "policy_calls": subtask_calls,
                    "student_sample_nfe": subtask_calls * PROTOCOL.student_steps,
                }
            )
            if not succeeded:
                break
            completed += 1
    except Exception as error:
        return {
            "schema": ROW_SCHEMA,
            "sequence_index": row["sequence_index"],
            "identity_sha256": row["identity_sha256"],
            "initial_state": row["initial_state"],
            "subtasks": row["subtasks"],
            "completed_prefix_length": completed,
            "primitive_steps": primitive_steps,
            "policy_calls": policy_calls,
            "student_sample_nfe": policy_calls * PROTOCOL.student_steps,
            "student_executed_calls": policy_calls * PROTOCOL.student_steps,
            "subtask_rows": subtask_rows,
            "noise_evidence_first_last": noise_evidence,
            "wall_seconds": time.perf_counter() - started,
            "exception": f"{type(error).__name__}: {error}",
        }
    return {
        "schema": ROW_SCHEMA,
        "sequence_index": row["sequence_index"],
        "identity_sha256": row["identity_sha256"],
        "initial_state": row["initial_state"],
        "subtasks": row["subtasks"],
        "completed_prefix_length": completed,
        "primitive_steps": primitive_steps,
        "policy_calls": policy_calls,
        "student_sample_nfe": policy_calls * PROTOCOL.student_steps,
        "student_executed_calls": policy_calls * PROTOCOL.student_steps,
        "subtask_rows": subtask_rows,
        "noise_evidence_first_last": noise_evidence,
        "wall_seconds": time.perf_counter() - started,
        "exception": None,
    }


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_existing(
    path: Path,
    assigned_ids: set[str],
    *,
    expected_metadata: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identity = row.get("identity_sha256")
        if identity not in assigned_ids or identity in rows or row.get("exception") is not None:
            raise ContractError(f"invalid resumable row in {path}: {identity}")
        mismatches = {
            key: (row.get(key), value)
            for key, value in expected_metadata.items()
            if row.get(key) != value
        }
        if mismatches:
            raise ContractError(f"resumable row identity mismatch in {path}: {mismatches}")
        rows[identity] = row
    return rows


def _load_oracle_and_annotations(path: Path) -> tuple[Any, Mapping[str, Any]]:
    """Load a caller-provided annotation/oracle bundle.

    The preferred public artifact is a YAML/JSON mapping with ``annotations``
    and ``task_oracle`` (a Hydra instantiation config).  For convenience, a
    raw annotation mapping is accepted when the same file has a sibling
    ``new_playtable_tasks.yaml`` or ``tasks/new_playtable_tasks.yaml``.
    """

    import hydra
    from omegaconf import OmegaConf

    loaded = OmegaConf.load(path)
    payload = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(payload, Mapping):
        raise ContractError("task-oracle-annotations must load as a mapping")
    annotations = payload.get("annotations", payload)
    oracle_config = payload.get("task_oracle", payload.get("oracle"))
    if oracle_config is None:
        candidates = (
            path.parent / "new_playtable_tasks.yaml",
            path.parent / "tasks" / "new_playtable_tasks.yaml",
        )
        for candidate in candidates:
            if candidate.is_file():
                oracle_config = OmegaConf.to_container(OmegaConf.load(candidate), resolve=True)
                break
    if not isinstance(annotations, Mapping) or oracle_config is None:
        raise ContractError(
            "task-oracle-annotations must provide annotations plus a Hydra task_oracle config"
        )
    return hydra.utils.instantiate(oracle_config), annotations


def _worker(
    args: argparse.Namespace,
    paths: RuntimePaths,
    panel: Mapping[str, Any],
) -> dict[str, Any]:
    validate_rlinf_host_support(
        paths.rlinf_checkout,
        benchmark="calvin_abc_d",
        patch_path=Path(__file__).resolve().parents[2]
        / "integrations"
        / "rlinf"
        / "path-opd-host-support.patch",
        require_openpi_distribution=True,
    )
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if world_size != args.world_size or rank < 0 or local_rank < 0 or rank != local_rank:
        raise RuntimeError(
            "Official-D evaluation must use one-node torchrun ranks 0..7; "
            f"got rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("real CALVIN evaluation requires CUDA; use --dry-run for CPU validation")
    device = _device_for_rank(torch, args.device, local_rank)
    if device.type != "cuda":
        raise RuntimeError("CALVIN simulator evaluation requires a CUDA device")
    torch.cuda.set_device(device)
    os.environ["EGL_VISIBLE_DEVICES"] = str(local_rank)
    dist.init_process_group("nccl")
    assigned = rows_for_rank(panel, rank, world_size)
    assigned_ids = {str(row["identity_sha256"]) for row in assigned}
    if len(assigned_ids) != len(assigned):
        raise ContractError("assigned Official-D rows contain duplicate identities")
    output = paths.output
    output.mkdir(parents=True, exist_ok=True)
    final = output / f"rank_{rank:03d}.jsonl"
    partial = output / f"rank_{rank:03d}.partial.jsonl"
    resumed_from_final = final.is_file()
    checkpoint_sha256 = sha256_file(checkpoint_artifact(paths.checkpoint))
    normalizer_sha256 = sha256_file(paths.normalization_stats)
    row_metadata = {
        "checkpoint_sha256": checkpoint_sha256,
        "normalization_stats_sha256": normalizer_sha256,
        "panel_sha256": str(panel["sha256"]),
        "method": args.method,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "horizon": args.horizon,
    }
    existing = _load_existing(
        final if resumed_from_final else partial,
        assigned_ids,
        expected_metadata=row_metadata,
    )
    environment: CalvinEnvironmentAdapter | None = None
    started = time.perf_counter()
    try:
        _seed_all(args.evaluation_seed + rank)
        model = _load_policy(paths, device)
        parameter_before = _parameter_sha256(model)
        task_oracle, annotations = _load_oracle_and_annotations(paths.task_oracle_annotations)
        environment = CalvinEnvironmentAdapter(
            rlinf_checkout=paths.rlinf_checkout,
            environment_assets=paths.environment_assets,
            rank=rank,
            world_size=world_size,
            seed=args.evaluation_seed,
            evaluation=True,
        )
        for row in assigned:
            identity = str(row["identity_sha256"])
            if identity in existing:
                continue
            result = evaluate_sequence(
                environment,
                model,
                task_oracle,
                annotations,
                row,
                device=device,
                evaluation_seed=args.evaluation_seed,
                horizon=args.horizon,
            )
            result.update(
                {"rank": rank, **row_metadata}
            )
            if result.get("exception") is not None:
                raise RuntimeError(f"row {row['sequence_index']} failed: {result['exception']}")
            _append_jsonl(partial, result)
            existing[identity] = result
        if len(existing) != len(assigned):
            raise RuntimeError("rank did not complete all assigned Official-D rows")
        if not resumed_from_final:
            os.replace(partial, final)
        parameter_after = _parameter_sha256(model)
        if parameter_before != parameter_after:
            raise RuntimeError("model parameter hash changed during evaluation")
        gradient_tensors = sum(parameter.grad is not None for parameter in model.parameters())
        if gradient_tensors:
            raise RuntimeError(f"evaluation materialized {gradient_tensors} gradient tensors")
        report = {
            "schema": "path-opd-calvin-evaluation-rank-v1",
            "status": "PASS",
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "assigned_rows": len(assigned),
            "completed_rows": len(existing),
            "parameter_sha256_before": parameter_before,
            "parameter_sha256_after": parameter_after,
            "parameter_hash_unchanged": True,
            "gradient_tensors": gradient_tensors,
            "optimizer_updates": 0,
            "backward_calls": 0,
            "row_file": str(final),
            "row_file_sha256": sha256_file(final),
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(output / f"rank_{rank:03d}_report.json", report)
        dist.barrier()
        return report
    finally:
        if environment is not None:
            environment.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _read_rows(
    output: Path, *, expected_metadata: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output.glob("rank_[0-9][0-9][0-9].jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                mismatches = {
                    key: (row.get(key), value)
                    for key, value in expected_metadata.items()
                    if row.get(key) != value
                }
                if mismatches:
                    raise ContractError(f"row identity mismatch in {path}: {mismatches}")
                rows.append(row)
    return rows


def summarize_rows(
    rows: list[Mapping[str, Any]],
    *,
    panel: Mapping[str, Any],
    checkpoint_sha256: str,
    normalizer_sha256: str,
    method: str,
    training_seed: int,
    evaluation_seed: int,
    horizon: int,
) -> dict[str, Any]:
    expected_rows = list(panel["rows"])
    expected_ids = {str(row["identity_sha256"]) for row in expected_rows}
    seen: dict[str, Mapping[str, Any]] = {}
    duplicate_rows = 0
    for row in rows:
        identity = str(row.get("identity_sha256"))
        if identity in seen:
            duplicate_rows += 1
        else:
            seen[identity] = row
    valid = [row for identity, row in seen.items() if identity in expected_ids]
    exceptions = [row for row in valid if row.get("exception") is not None]
    completed = [
        int(row.get("completed_prefix_length", 0)) for row in valid if row.get("exception") is None
    ]
    denominator = len(expected_rows)
    metrics = {
        "average_completed_sequence_length": sum(completed) / denominator if denominator else 0.0,
        **{
            f"SR{index}": sum(value >= index for value in completed) / denominator
            if denominator
            else 0.0
            for index in range(1, PROTOCOL.sequence_length + 1)
        },
    }
    formal_benchmark = bool(panel.get("formal_panel", False)) and (
        horizon == PROTOCOL.evaluation_subtask_limit
    )
    missing = expected_ids - set(seen)
    status = (
        "COMPLETE"
        if len(valid) == denominator and not exceptions and not missing and duplicate_rows == 0
        else "FAILED"
    )
    return {
        "schema": SCHEMA,
        "status": status,
        "benchmark": PROTOCOL.benchmark,
        "method": method,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "horizon": horizon,
        # ``status`` describes row completion. These fields describe whether
        # the rows are eligible to be interpreted as the formal Official-D
        # metric.
        "formal_benchmark": formal_benchmark,
        "non_formal": not formal_benchmark,
        "qualification": not formal_benchmark,
        "protocol": paper_contract()["evaluation"],
        "coverage": {
            "expected_rows": denominator,
            "actual_rows": len(rows),
            "unique_identity_sha256": len(seen),
            "missing_rows": len(missing),
            "duplicate_rows": duplicate_rows,
            "exception_rows": len(exceptions),
        },
        "metrics": metrics,
        "checkpoint_sha256": checkpoint_sha256,
        "normalization_stats_sha256": normalizer_sha256,
        "panel_sha256": panel.get("sha256"),
        "rows": rows,
        "private_paths_omitted": True,
    }


def _dry_run_report(args: argparse.Namespace, paths: RuntimePaths) -> dict[str, Any]:
    panel = None
    if args.panel.is_file():
        panel = validate_panel(args.panel, allow_custom=args.allow_custom_panel)
    report = dry_run_plan(
        "evaluate",
        paths,
        method=args.method,
        seed=args.training_seed,
        target_chunks=PROTOCOL.target_chunks,
        max_updates=None,
        panel=panel,
    )
    report["schema"] = SCHEMA
    report["evaluation"].update(
        {
            "method": args.method,
            "training_seed": args.training_seed,
            "evaluation_seed": args.evaluation_seed,
            "world_size": args.world_size,
            "horizon": args.horizon,
            "inference_only": True,
            "checkpoint_manifest_checked": False,
            "panel_path": str(args.panel),
        }
    )
    report["output_schema"] = RESULT_SCHEMA
    report["runtime"] = {
        "launcher": (
            "torchrun --standalone --nproc-per-node 8 benchmarks/calvin_abc_d/evaluate.py ..."
        ),
        "imports_deferred_until_real_run": [
            "torch",
            "omegaconf",
            "hydra",
            "rlinf",
            "calvin_agent",
            "pybullet",
        ],
        "dry_run_performs_evaluation": False,
    }
    return report


def _synthetic_smoke(argv: list[str]) -> int:
    """Run the dependency-light train/checkpoint/panel/trace evaluator proof."""

    parser = argparse.ArgumentParser(
        prog="calvin-evaluate --synthetic-smoke",
        description=(
            "Run the synthetic CALVIN evaluator contract; no RLinf checkout, "
            "model, panel, simulator, or CUDA device is used."
        ),
    )
    parser.add_argument("--synthetic-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("artifacts/evaluator-smoke/calvin_abc_d")
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
        "calvin_abc_d",
        args.work_dir,
        entrypoint="benchmarks/calvin_abc_d/evaluate.py",
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
            print(f"calvin-evaluate synthetic smoke: {error}", file=sys.stderr)
            return 2
    args = build_parser().parse_args(raw_argv)
    try:
        paths = _validate_args(args)
        if args.dry_run:
            report = _dry_run_report(args, paths)
            rendered = json.dumps(report, indent=2, sort_keys=True)
            if args.contract_output is not None:
                atomic_write_json(args.contract_output, report)
            else:
                print(rendered)
            return 0
        panel = validate_panel(args.panel, allow_custom=args.allow_custom_panel)
        report = _worker(args, paths, panel)
        rank = int(os.environ.get("RANK", "-1"))
        if rank == 0:
            checkpoint_sha256 = sha256_file(checkpoint_artifact(paths.checkpoint))
            normalizer_sha256 = sha256_file(paths.normalization_stats)
            row_metadata = {
                "checkpoint_sha256": checkpoint_sha256,
                "normalization_stats_sha256": normalizer_sha256,
                "panel_sha256": str(panel["sha256"]),
                "method": args.method,
                "training_seed": args.training_seed,
                "evaluation_seed": args.evaluation_seed,
                "horizon": args.horizon,
            }
            rows = _read_rows(paths.output, expected_metadata=row_metadata)
            summary = summarize_rows(
                rows,
                panel=panel,
                checkpoint_sha256=checkpoint_sha256,
                normalizer_sha256=normalizer_sha256,
                method=args.method,
                training_seed=args.training_seed,
                evaluation_seed=args.evaluation_seed,
                horizon=args.horizon,
            )
            atomic_write_json(paths.output / "results.json", summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report.get("status") == "PASS" else 2
    except (ContractError, ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
        failure = {
            "schema": SCHEMA,
            "status": "FAIL_CLOSED",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

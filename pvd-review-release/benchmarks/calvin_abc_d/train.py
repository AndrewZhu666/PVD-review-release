#!/usr/bin/env python3
"""Train the CALVIN ABC-D Endpoint-DAgger or Path-OPD arm.

The command is intentionally parameterized.  A reviewer supplies an RLinf
checkout, both model directories, CALVIN assets, annotations, and an output
directory.  ``--dry-run`` validates the contract without importing CUDA,
PyBullet, OpenPI, or the simulator.  Real training is launched with
``torchrun --nproc-per-node 8`` and writes resumable checkpoints at the
recorded fresh-chunk boundaries.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
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
    build_checkpoint_manifest,
    git_revision,
    tree_sha256,
    validate_rlinf_host_support,
)
from path_opd.release_contract import (  # noqa: E402
    validate_checkpoint_manifest as validate_release_checkpoint_manifest,
)

try:
    from .common import (
        CLIP_RESOLUTION,
        OPTIMIZER,
        PROTOCOL,
        SAFE_BOUNDARY,
        ContractError,
        RuntimePaths,
        activate_rlinf,
        asset_fingerprint,
        atomic_write_json,
        checkpoint_artifact,
        dry_run_plan,
        load_checkpoint_manifest,
        paper_contract,
        sha256_file,
        validate_checkpoint_manifest,
        validate_train_paths,
    )
    from .environment import CalvinEnvironmentAdapter, policy_observation, prepare_actions
except ImportError:  # direct ``python benchmarks/calvin_abc_d/train.py`` invocation
    from benchmarks.calvin_abc_d.common import (  # type: ignore[no-redef]
        CLIP_RESOLUTION,
        OPTIMIZER,
        PROTOCOL,
        SAFE_BOUNDARY,
        ContractError,
        RuntimePaths,
        activate_rlinf,
        asset_fingerprint,
        atomic_write_json,
        checkpoint_artifact,
        dry_run_plan,
        load_checkpoint_manifest,
        paper_contract,
        sha256_file,
        validate_checkpoint_manifest,
        validate_train_paths,
    )
    from benchmarks.calvin_abc_d.environment import (  # type: ignore[no-redef]
        CalvinEnvironmentAdapter,
        policy_observation,
        prepare_actions,
    )


SCHEMA = "path-opd-calvin-train-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("path_opd", "endpoint_dagger"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--world-size", type=int, default=PROTOCOL.world_size)
    parser.add_argument("--target-chunks", type=int, default=PROTOCOL.target_chunks)
    parser.add_argument(
        "--max-updates", type=int, help="qualification cap; one update is a smoke run"
    )
    parser.add_argument("--rlinf-checkout", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument(
        "--normalization-stats", "--norm", dest="normalization_stats", type=Path, required=True
    )
    parser.add_argument("--environment-assets", type=Path, required=True)
    parser.add_argument("--task-oracle-annotations", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_resume_checkpoint_digest(
    artifact: Path, manifest: Mapping[str, Any]
) -> None:
    expected = manifest.get("checkpoint_sha256")
    if expected != sha256_file(artifact):
        raise ContractError("resume checkpoint file digest does not match manifest")


def _validate_args(args: argparse.Namespace) -> RuntimePaths:
    if args.world_size != PROTOCOL.world_size:
        raise ContractError(f"CALVIN training requires world_size={PROTOCOL.world_size}")
    if args.target_chunks <= 0 or args.target_chunks % PROTOCOL.world_size:
        raise ContractError("target-chunks must be a positive multiple of eight")
    planned_updates = args.target_chunks // PROTOCOL.world_size
    if args.max_updates is not None and not 1 <= args.max_updates <= planned_updates:
        raise ContractError("max-updates must be between one and target-chunks / eight")
    paths = validate_train_paths(
        rlinf_checkout=args.rlinf_checkout,
        base_model=args.base_model,
        frozen_teacher=args.teacher_model,
        normalization_stats=args.normalization_stats,
        environment_assets=args.environment_assets,
        task_oracle_annotations=args.task_oracle_annotations,
        output=args.output,
        require_existing=not args.dry_run,
    )
    if args.resume is not None:
        manifest = load_checkpoint_manifest(args.resume)
        artifact = checkpoint_artifact(args.resume)
        _validate_resume_checkpoint_digest(artifact, manifest)
        normalizer_hash = (
            sha256_file(paths.normalization_stats) if paths.normalization_stats.is_file() else None
        )
        validate_checkpoint_manifest(
            manifest,
            method=args.method,
            seed=args.seed,
            world_size=args.world_size,
            normalizer_sha256=normalizer_hash,
        )
        try:
            validate_release_checkpoint_manifest(
                manifest["release_contract"],
                expected={
                    "benchmark": PROTOCOL.benchmark,
                    "method": args.method,
                    "seed": args.seed,
                    "world_size": args.world_size,
                    "contract": paper_contract(),
                },
                assets=_release_asset_hashes(paths),
                code=_release_code_provenance(paths),
            )
        except (KeyError, ReleaseContractError) as error:
            raise ContractError(f"resume release contract mismatch: {error}") from error
        if args.world_size > 1 and manifest.get("rank_state_count") != args.world_size:
            raise ContractError(
                "distributed CALVIN resume requires a checkpoint with one environment/RNG "
                "state per rank"
            )
        args.resume_manifest = manifest
    return paths


def _release_asset_hashes(paths: RuntimePaths) -> dict[str, str]:
    values = {
        "base_model": paths.base_model,
        "frozen_teacher": paths.frozen_teacher,
        "normalization_stats": paths.normalization_stats,
        "environment_assets": paths.environment_assets,
        "task_oracle_annotations": paths.task_oracle_annotations,
    }
    return {name: tree_sha256(path) for name, path in values.items() if path is not None}


def _release_code_provenance(paths: RuntimePaths) -> dict[str, str | None]:
    return {
        "release_source_sha256": tree_sha256(
            _REPOSITORY_ROOT, include=("src", "benchmarks", "configs")
        ),
        "rlinf_revision": git_revision(paths.rlinf_checkout),
    }


def _release_checkpoint_identity(
    args: argparse.Namespace,
    paths: RuntimePaths,
    *,
    update: int,
) -> dict[str, Any]:
    return build_checkpoint_manifest(
        benchmark=PROTOCOL.benchmark,
        method=args.method,
        seed=args.seed,
        world_size=args.world_size,
        update=update,
        contract=paper_contract(),
        assets=_release_asset_hashes(paths),
        code=_release_code_provenance(paths),
        extra={"target_chunks": args.target_chunks},
    )


def _model_config(path: Path, normalizer: Path, *, train_expert_only: bool) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "model_path": str(path),
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
                "train_expert_only": train_expert_only,
                "add_value_head": False,
                "value_after_vlm": False,
                "is_nft": False,
            },
        }
    )


def _load_model(path: Path, normalizer: Path, *, train_expert_only: bool, device: Any) -> Any:
    from rlinf.models.embodiment.openpi import get_model

    model = get_model(_model_config(path, normalizer, train_expert_only=train_expert_only))
    model.to(device)
    if not train_expert_only:
        model.requires_grad_(False)
    model.eval()
    return model


def _module_sha256(model: Any) -> str:
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


def _tree_cpu(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _tree_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_cpu(item) for item in value)
    return value


def _capture_rng(device: Any) -> dict[str, Any]:
    """Capture every rank-local RNG used after a checkpoint boundary."""

    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }


def _restore_rng(state: Mapping[str, Any], device: Any) -> None:
    """Restore the rank-local RNG captured by :func:`_capture_rng`."""

    import numpy as np
    import torch

    required = ("python", "numpy", "torch", "cuda")
    missing = [name for name in required if name not in state]
    if missing:
        raise ContractError("checkpoint RNG state is missing: " + ", ".join(missing))
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)


def _rank_state_for_resume(
    payload: Mapping[str, Any], *, rank: int, world_size: int
) -> Mapping[str, Any]:
    """Select one rank's environment/RNG continuation from a checkpoint.

    Older CALVIN checkpoints stored only rank zero's continuation.  They are
    accepted for a single-rank diagnostic, but fail closed for the recorded
    eight-rank protocol instead of silently restoring every worker to the same
    simulator state.
    """

    raw_states = payload.get("rank_states")
    if raw_states is None:
        if world_size != 1:
            raise ContractError(
                "checkpoint lacks per-rank continuation state; distributed CALVIN resume "
                "cannot be claimed"
            )
        continuation = payload.get("continuation")
        if not isinstance(continuation, Mapping):
            raise ContractError("checkpoint lacks environment continuation state")
        legacy_rng = payload.get("rng")
        return {
            "rank": 0,
            "continuation": continuation,
            "rng": legacy_rng,
        }
    if not isinstance(raw_states, list) or len(raw_states) != world_size:
        raise ContractError(
            "checkpoint rank_states must contain exactly one state per distributed rank"
        )
    selected: Mapping[str, Any] | None = None
    seen: set[int] = set()
    for item in raw_states:
        if not isinstance(item, Mapping) or not isinstance(item.get("rank"), int):
            raise ContractError("checkpoint rank state has an invalid rank id")
        item_rank = int(item["rank"])
        if item_rank in seen or not 0 <= item_rank < world_size:
            raise ContractError("checkpoint rank states have duplicate or out-of-range ranks")
        seen.add(item_rank)
        if item_rank == rank:
            selected = item
    if seen != set(range(world_size)) or selected is None:
        raise ContractError("checkpoint rank states do not cover the launched world")
    continuation = selected.get("continuation")
    if not isinstance(continuation, Mapping):
        raise ContractError("checkpoint rank state lacks environment continuation")
    rng = selected.get("rng")
    if not isinstance(rng, Mapping):
        raise ContractError("checkpoint rank state lacks RNG continuation")
    return selected


def _bool_sequence(value: Any) -> list[bool]:
    """Flatten tensor/array/list termination flags without importing NumPy."""

    if value is None:
        return []
    if isinstance(value, bool):
        return [value]
    # Torch tensors expose ``detach``/``tolist``; NumPy arrays expose
    # ``tolist``.  Converting before recursing avoids the ambiguous truth-value
    # error raised by multi-element arrays.
    if hasattr(value, "detach"):
        with contextlib.suppress(AttributeError, RuntimeError):
            value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        with contextlib.suppress(AttributeError, ValueError):
            value = value.tolist()
    if isinstance(value, Mapping):
        return [item for nested in value.values() for item in _bool_sequence(nested)]
    if isinstance(value, (list, tuple)):
        return [item for nested in value for item in _bool_sequence(nested)]
    try:
        return [bool(value)]
    except (TypeError, ValueError):
        return []


def _chunk_length(actions: Any, observations: Any) -> int:
    """Infer the number of primitive actions represented by one chunk."""

    shape = getattr(actions, "shape", None)
    if shape is not None:
        try:
            dimensions = tuple(int(item) for item in shape)
            if len(dimensions) >= 3:
                return max(0, dimensions[-2])
            if len(dimensions) == 2:
                return max(0, dimensions[0])
        except (TypeError, ValueError):
            pass
    if isinstance(observations, (list, tuple)):
        return len(observations)
    return 0


def _chunk_progress(
    actions: Any,
    observations: Any,
    terminations: Any,
    truncations: Any,
) -> tuple[int, bool]:
    """Return executed primitive steps and whether the episode ended.

    RLinf's vector wrapper returns termination matrices shaped ``[env, chunk]``;
    the raw fallback returns flat lists.  This helper handles both and stops the
    count at the first terminal flag so an overlong final chunk cannot inflate
    the episode cursor.
    """

    length = _chunk_length(actions, observations)
    termination_flags = _bool_sequence(terminations)
    truncation_flags = _bool_sequence(truncations)
    flags = [
        (termination_flags[index] if index < len(termination_flags) else False)
        or (truncation_flags[index] if index < len(truncation_flags) else False)
        for index in range(max(len(termination_flags), len(truncation_flags)))
    ]
    done = any(flags)
    if done:
        with contextlib.suppress(ValueError):
            length = min(length, flags.index(True) + 1)
    return length, done


def _training_reset_due(
    *,
    episode_steps: int,
    chunks_since_reset: int,
    terminal: bool,
) -> bool:
    """Apply the declared episode and bounded-history reset policy."""

    return bool(
        terminal
        or episode_steps >= PROTOCOL.training_episode_limit
        or chunks_since_reset >= PROTOCOL.reset_interval_chunks
    )


def _seed_all(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fixed_noise(device: Any, seed: int, update: int) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed * 1_000_003 + update * 97 + 2_026_0819))
    return torch.randn(
        (1, PROTOCOL.model_horizon, PROTOCOL.model_action_dims),
        generator=generator,
        dtype=torch.float32,
    ).to(device)


def _rollout(
    model: Any,
    observation: Mapping[str, Any],
    *,
    noise: Any,
    device: Any,
    inference_only: bool = False,
) -> tuple[Any, dict[str, Any]]:
    import torch

    policy_input = policy_observation(observation)
    policy_input = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in policy_input.items()
    }
    context = torch.no_grad() if inference_only else contextlib.nullcontext()
    with context:
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
        raise RuntimeError("CALVIN K8 endpoint does not equal model_actions")
    if tuple(actions.shape) != (1, PROTOCOL.execution_prefix, PROTOCOL.physical_action_dims):
        raise RuntimeError(f"CALVIN physical action shape drift: {tuple(actions.shape)}")
    return actions, result


def _dagger_loss(
    student: Any,
    student_result: Mapping[str, Any],
    teacher_result: Mapping[str, Any],
) -> Any:
    from path_opd.adapters.openpi import OpenPIAdapter

    adapter = OpenPIAdapter.from_rollout(
        student,
        student_result,
        steps=PROTOCOL.student_steps,
        nft_forward_type=_PathRuntime._forward_enum().NFT,
        sft_forward_type=_PathRuntime._forward_enum().SFT,
    )
    return adapter.endpoint_dagger_loss(teacher_result["model_actions"])


class _PathRuntime:
    """Keep one immutable teacher identity while rebinding rollout context."""

    def __init__(self, teacher: Any) -> None:
        from path_opd import ActionContract, FlowSchedule, PathOPD
        from path_opd.core import FrozenTeacher

        self._forward_inputs: Mapping[str, Any] | None = None
        self._forward_type = self._forward_enum().NFT
        self.objective = PathOPD(
            ActionContract(PROTOCOL.execution_prefix, PROTOCOL.physical_action_dims),
            FlowSchedule.uniform(PROTOCOL.student_steps),
        )

        def query(model: Any, states: Any, times: Any) -> Any:
            from path_opd.adapters.openpi import OpenPIAdapter

            if self._forward_inputs is None:
                raise RuntimeError("Path-OPD teacher context is not bound")
            return OpenPIAdapter(
                model=model,
                forward_inputs=self._forward_inputs,
                steps=PROTOCOL.teacher_steps,
                nft_forward_type=self._forward_type,
                contract=ActionContract(PROTOCOL.execution_prefix, PROTOCOL.physical_action_dims),
            ).velocity(states, times)

        self.teacher = FrozenTeacher(teacher, query, deep_fingerprint=False)

    @staticmethod
    def _forward_enum() -> Any:
        from rlinf.models.embodiment.base_policy import ForwardType

        return ForwardType

    def loss(self, student: Any, result: Mapping[str, Any]) -> Any:
        from path_opd.adapters.openpi import OpenPIAdapter

        forward_inputs = result.get("forward_inputs")
        if not isinstance(forward_inputs, Mapping):
            raise RuntimeError("OpenPI rollout lacks forward_inputs")
        self._forward_inputs = forward_inputs
        adapter = OpenPIAdapter.from_rollout(
            student,
            result,
            steps=PROTOCOL.student_steps,
            nft_forward_type=self._forward_type,
            contract=self.objective.contract,
        )
        return self.objective.supervise(adapter.chains, adapter.velocity, self.teacher).loss


def _allreduce_grads(model: Any, names: tuple[str, ...], world_size: int) -> None:
    import torch
    import torch.distributed as dist

    parameters = dict(model.named_parameters())
    for name in names:
        parameter = parameters[name]
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"missing/nonfinite action-expert gradient: {name}")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def _checkpoint(
    *,
    paths: RuntimePaths,
    args: argparse.Namespace,
    model: Any,
    optimizer: Any,
    trainable_names: tuple[str, ...],
    teacher_hash: str,
    update: int,
    chunks: int,
    continuation: Mapping[str, Any],
    rank_states: list[Mapping[str, Any]],
) -> Path:
    import torch

    output = paths.output / "checkpoints" / f"chunks_{chunks:06d}"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "trainer_state.pt"
    model_hash = _module_sha256(model)
    payload = {
        "model": _tree_cpu(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "chunks": chunks,
        "continuation": _tree_cpu(continuation),
        "rank_states": _tree_cpu(rank_states),
        "rank_state_count": len(rank_states),
        "safe_boundary": SAFE_BOUNDARY,
    }
    temporary = state_path.with_name(f".{state_path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, state_path)
    manifest = {
        "schema_version": 1,
        "benchmark": PROTOCOL.benchmark,
        "method": args.method,
        "seed": args.seed,
        "world_size": args.world_size,
        "student_steps": PROTOCOL.student_steps,
        "teacher_steps": PROTOCOL.teacher_steps,
        "model_horizon": PROTOCOL.model_horizon,
        "execution_prefix": PROTOCOL.execution_prefix,
        "physical_action_dims": PROTOCOL.physical_action_dims,
        "model_action_dims": PROTOCOL.model_action_dims,
        "safe_boundary": SAFE_BOUNDARY,
        "chunks": chunks,
        "optimizer_updates": update,
        "rank_state_count": len(rank_states),
        "distributed_resume_state": "per_rank_environment_and_rng",
        "normalization_stats_sha256": sha256_file(paths.normalization_stats),
        "teacher_sha256": teacher_hash,
        "student_sha256": model_hash,
        "checkpoint_sha256": sha256_file(state_path),
        "trainable_parameter_names_sha256": hashlib.sha256(
            "\n".join(trainable_names).encode()
        ).hexdigest(),
        "clip_resolution": CLIP_RESOLUTION,
        "assets": {
            "base_model": asset_fingerprint(paths.base_model),
            "frozen_teacher": asset_fingerprint(paths.frozen_teacher),
            "normalization_stats": asset_fingerprint(paths.normalization_stats),
            "environment_assets": asset_fingerprint(paths.environment_assets),
            "task_oracle_annotations": asset_fingerprint(paths.task_oracle_annotations),
        },
        "private_paths_omitted": True,
    }
    manifest["release_contract"] = _release_checkpoint_identity(args, paths, update=update)
    atomic_write_json(output / "manifest.json", manifest)
    # Read back immediately so a partial/unsafe save is visible at the source.
    loaded = torch.load(state_path, map_location="cpu", weights_only=False)
    if (
        loaded.get("update") != update
        or loaded.get("chunks") != chunks
        or loaded.get("rank_state_count") != len(rank_states)
    ):
        raise RuntimeError("checkpoint save/reload cursor mismatch")
    return output


def _run(args: argparse.Namespace, paths: RuntimePaths) -> int:
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

    from path_opd import configure_action_expert

    activate_rlinf(paths.rlinf_checkout)
    if not torch.cuda.is_available():
        raise RuntimeError("CALVIN training requires CUDA; use --dry-run for CPU validation")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world = int(os.environ.get("WORLD_SIZE", "-1"))
    if rank < 0 or local_rank < 0 or world != args.world_size:
        raise RuntimeError(
            f"launch with torchrun and {args.world_size} ranks; got {rank}/{local_rank}/{world}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    environment: CalvinEnvironmentAdapter | None = None
    try:
        if rank == 0:
            if args.resume is None and paths.output.exists() and any(paths.output.iterdir()):
                raise ContractError(
                    "output already contains files; choose a new directory or --resume"
                )
            paths.output.mkdir(parents=True, exist_ok=True)
            (paths.output / "logs").mkdir(exist_ok=True)
        dist.barrier()
        _seed_all(args.seed + rank * PROTOCOL.rank_seed_multiplier)
        student = _load_model(
            paths.base_model, paths.normalization_stats, train_expert_only=True, device=device
        )
        trainable_names = tuple(configure_action_expert(student))
        teacher = _load_model(
            paths.frozen_teacher, paths.normalization_stats, train_expert_only=False, device=device
        )
        teacher_hash = _module_sha256(teacher)
        path_runtime = _PathRuntime(teacher) if args.method == "path_opd" else None
        optimizer = torch.optim.AdamW(
            [parameter for parameter in student.parameters() if parameter.requires_grad],
            lr=OPTIMIZER["learning_rate"],
            betas=tuple(OPTIMIZER["betas"]),
            eps=OPTIMIZER["epsilon"],
            weight_decay=OPTIMIZER["weight_decay"],
        )
        environment = CalvinEnvironmentAdapter(
            rlinf_checkout=paths.rlinf_checkout,
            environment_assets=paths.environment_assets,
            rank=rank,
            world_size=world,
            seed=args.seed,
        )
        observation = environment.reset()
        update = 0
        chunks = 0
        action_history: list[Any] = []
        episode_steps = 0
        chunks_since_reset = 0
        reset_pending = False
        if args.resume is not None:
            artifact = checkpoint_artifact(args.resume)
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            student.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            update = int(payload["update"])
            chunks = int(payload["chunks"])
            rank_state = _rank_state_for_resume(payload, rank=rank, world_size=world)
            continuation = rank_state["continuation"]
            observation = environment.restore_continuation(continuation)
            action_history = list(continuation.get("action_history_since_reset", []))
            if len(action_history) > PROTOCOL.reset_interval_chunks:
                raise ContractError(
                    "checkpoint action history exceeds the bounded reset interval"
                )
            chunks_since_reset = int(
                continuation.get("chunks_since_reset", len(action_history))
            )
            episode_steps = int(
                continuation.get(
                    "primitive_steps_since_reset",
                    chunks_since_reset * PROTOCOL.execution_prefix,
                )
            )
            reset_pending = bool(continuation.get("reset_pending", False))
            if chunks_since_reset < 0 or episode_steps < 0:
                raise ContractError("checkpoint contains negative CALVIN episode counters")
            rng_state = rank_state.get("rng")
            if not isinstance(rng_state, Mapping):
                if world != 1:
                    raise ContractError(
                        "distributed CALVIN checkpoint lacks rank-local RNG state"
                    )
            else:
                # Restore before replaying a pending reset so any random draws
                # made by that reset occur at the same point as uninterrupted
                # training after the checkpoint boundary.
                _restore_rng(rng_state, device)
            if reset_pending:
                observation = environment.reset()
                action_history.clear()
                episode_steps = 0
                chunks_since_reset = 0
                reset_pending = False
        target_updates = args.target_chunks // args.world_size
        if args.max_updates is not None:
            target_updates = args.max_updates
        while update < target_updates:
            update += 1
            noise = _fixed_noise(device, args.seed + rank, update)
            student.eval()
            student_actions, student_result = _rollout(
                student,
                observation,
                noise=noise,
                device=device,
                inference_only=False,
            )
            teacher_result = None
            if args.method == "endpoint_dagger":
                _, teacher_result = _rollout(
                    teacher,
                    observation,
                    noise=noise,
                    device=device,
                    inference_only=True,
                )
            physical_actions = prepare_actions(student_actions)
            next_observations, _rewards, _terms, _truncs, _infos = environment.chunk_step(
                physical_actions
            )
            try:
                has_next_observation = len(next_observations) > 0
            except TypeError:
                has_next_observation = False
            next_observation = (
                next_observations[-1] if has_next_observation else environment.get_obs()
            )
            primitive_steps, terminal = _chunk_progress(
                physical_actions, next_observations, _terms, _truncs
            )
            episode_steps += primitive_steps
            chunks_since_reset += 1
            action_history.append(_tree_cpu(physical_actions))
            if len(action_history) > PROTOCOL.reset_interval_chunks:
                raise RuntimeError("CALVIN action history exceeded its bounded reset interval")
            optimizer.zero_grad(set_to_none=True)
            student.train()
            if args.method == "endpoint_dagger":
                if teacher_result is None:
                    raise AssertionError("teacher endpoint rollout is missing")
                loss = _dagger_loss(student, student_result, teacher_result)
            else:
                if path_runtime is None:
                    raise AssertionError("Path-OPD runtime is missing")
                loss = path_runtime.loss(student, student_result)
            if not torch.isfinite(loss).all():
                raise RuntimeError("non-finite CALVIN training loss")
            loss.backward()
            _allreduce_grads(student, trainable_names, world)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in student.parameters() if parameter.requires_grad],
                CLIP_RESOLUTION["max_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            chunks = update * world
            if rank == 0:
                with (paths.output / "logs" / "rank_00.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(
                        json.dumps(
                            {
                                "schema": SCHEMA,
                                "method": args.method,
                                "seed": args.seed,
                                "update": update,
                                "chunks": chunks,
                                "loss": float(loss.detach().cpu()),
                                "teacher_nfe": PROTOCOL.teacher_steps,
                                "teacher_calls": 1,
                                "clip_grad": CLIP_RESOLUTION["max_norm"],
                                "evaluation_executed": False,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            observation = next_observation
            reset_pending = _training_reset_due(
                episode_steps=episode_steps,
                chunks_since_reset=chunks_since_reset,
                terminal=terminal,
            )
            checkpoint_due = chunks in PROTOCOL.checkpoint_chunks or update == target_updates
            continuation = None
            if checkpoint_due:
                continuation = {
                    "environment_state": environment.capture_state(),
                    "observation": _tree_cpu(observation),
                    "action_history_since_reset": _tree_cpu(action_history),
                    "primitive_steps_since_reset": episode_steps,
                    "chunks_since_reset": chunks_since_reset,
                    "reset_pending": reset_pending,
                }
                local_rank_state = {
                    "rank": rank,
                    "continuation": continuation,
                    "rng": _capture_rng(device),
                }
                rank_states: list[Mapping[str, Any]]
                if world > 1:
                    gathered: list[Mapping[str, Any] | None] = [None] * world
                    dist.all_gather_object(gathered, _tree_cpu(local_rank_state))
                    if any(item is None for item in gathered):
                        raise ContractError(
                            "distributed CALVIN checkpoint gather returned an empty rank"
                        )
                    rank_states = [item for item in gathered if item is not None]
                else:
                    rank_states = [local_rank_state]
                if rank == 0:
                    _checkpoint(
                        paths=paths,
                        args=args,
                        model=student,
                        optimizer=optimizer,
                        trainable_names=trainable_names,
                        teacher_hash=teacher_hash,
                        update=update,
                        chunks=chunks,
                        continuation=continuation,
                        rank_states=rank_states,
                    )
                dist.barrier()
            if reset_pending:
                observation = environment.reset()
                action_history.clear()
                episode_steps = 0
                chunks_since_reset = 0
                reset_pending = False
        if _module_sha256(teacher) != teacher_hash:
            raise RuntimeError("frozen CALVIN teacher changed during training")
        if rank == 0:
            atomic_write_json(
                paths.output / "progress.json",
                {
                    "schema": SCHEMA,
                    "status": "QUALIFICATION_PASS"
                    if args.max_updates is not None
                    else "PAPER_RUN_COMPLETE",
                    "method": args.method,
                    "seed": args.seed,
                    "updates": update,
                    "chunks": chunks,
                    "paper_contract": paper_contract(),
                    "private_paths_omitted": True,
                },
            )
        dist.barrier()
    finally:
        if environment is not None:
            environment.close()
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


def _synthetic_smoke(argv: list[str]) -> int:
    """Run the synthetic CALVIN contract through the released worker entrypoint."""

    parser = argparse.ArgumentParser(
        prog="calvin-train --synthetic-smoke",
        description=(
            "Run the synthetic CALVIN runner contract; no RLinf checkout, "
            "model, simulator, CUDA device, or benchmark asset is used."
        ),
    )
    parser.add_argument("--synthetic-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("artifacts/runner-smoke/calvin_abc_d")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    from path_opd.adapters.benchmark_smoke import (
        SyntheticSmokeConfig,
        run_runner_synthetic_smoke,
        write_report,
    )

    report = run_runner_synthetic_smoke(
        "calvin_abc_d",
        args.work_dir,
        entrypoint="benchmarks/calvin_abc_d/train.py",
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
            print(f"calvin-train synthetic smoke: {error}", file=sys.stderr)
            return 2
    args = build_parser().parse_args(raw_argv)
    try:
        paths = _validate_args(args)
        if args.dry_run:
            print(
                json.dumps(
                    dry_run_plan(
                        "train",
                        paths,
                        method=args.method,
                        seed=args.seed,
                        target_chunks=args.target_chunks,
                        max_updates=args.max_updates,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        return _run(args, paths)
    except (ContractError, ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"calvin-train: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

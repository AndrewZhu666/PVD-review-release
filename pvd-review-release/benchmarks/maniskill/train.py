#!/usr/bin/env python3

# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Train ManiSkill Path-OPD or the matched Endpoint-DAgger baseline.

This release-facing worker is derived from Apache-2.0 RLinf training workers
whose source identities are recorded by digest in the release provenance
document. Machine supervisors, private paths,
accelerator-model checks, monitoring, and fixed private parameter digests were
intentionally removed.  The scientific K8/action/loss/exposure contract is
kept explicit and is executed through the released :mod:`path_opd` core.

Use ``torchrun`` for multi-rank execution.  A smaller ``--max-updates`` is a
qualification run; it is never labelled as a full paper-budget result.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

from path_opd.release_contract import (
    ReleaseContractError,
    build_checkpoint_manifest,
    git_revision,
    tree_sha256,
    validate_checkpoint_manifest,
    validate_rlinf_host_support,
)

if __package__:
    from .common import (
        CONTRACT,
        MANISKILL_CONTROL_MODE,
        ContractError,
        asset_fingerprint,
        paper_contract_dict,
        resolve_contract,
        sha256_file,
        validate_assets,
        validate_rlinf_checkout,
        write_json,
    )
else:  # pragma: no cover - exercised by subprocess CLI tests
    from common import (  # type: ignore[no-redef]
        CONTRACT,
        MANISKILL_CONTROL_MODE,
        ContractError,
        asset_fingerprint,
        paper_contract_dict,
        resolve_contract,
        sha256_file,
        validate_assets,
        validate_rlinf_checkout,
        write_json,
    )


SCHEMA = "path-opd-maniskill-train-v1"
CONTEXT_KEYS = (
    "observation/image",
    "observation/state",
    "tokenized_prompt",
    "tokenized_prompt_mask",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Portable RLinf/OpenPI ManiSkill K8 trainer",
    )
    parser.add_argument("--method", choices=CONTRACT.method_choices, required=True)
    parser.add_argument("--seed", type=int, default=0, choices=CONTRACT.supported_seeds)
    parser.add_argument("--world-size", type=int, default=CONTRACT.world_size)
    parser.add_argument(
        "--max-updates",
        type=int,
        default=CONTRACT.optimizer_updates,
        help="10,000 is the full 80k-chunk paper budget at world size 8",
    )
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument(
        "--teacher-normalization-stats",
        type=Path,
        default=None,
        help=(
            "normalization statistics for the frozen teacher; defaults to "
            "--normalization-stats for backward compatibility"
        ),
    )
    parser.add_argument(
        "--simulator-assets",
        type=Path,
        required=True,
        help="RLinf task assets (exported as MANISKILL_ASSET_DIR)",
    )
    parser.add_argument(
        "--maniskill-package-assets",
        "--maniskill-assets",
        dest="maniskill_package_assets",
        type=Path,
        required=True,
        help=(
            "ManiSkill package assets (exported as MS_ASSET_DIR); "
            "legacy alias: --maniskill-assets"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--contract-output",
        type=Path,
        help="write dry-run JSON here instead of standard output",
    )
    return parser


def teacher_normalization_stats(args: argparse.Namespace) -> Path:
    """Return the teacher normalizer, preserving the legacy CLI default."""

    value = getattr(args, "teacher_normalization_stats", None)
    return args.normalization_stats if value is None else value


def _asset_arguments(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "base_model": args.base_model,
        "frozen_teacher": args.teacher_model,
        "normalization_stats": args.normalization_stats,
        "teacher_normalization_stats": teacher_normalization_stats(args),
        "simulator_assets": args.simulator_assets,
        "maniskill_package_assets": args.maniskill_package_assets,
    }


def _resolved_report(args: argparse.Namespace, *, require_existing: bool) -> dict[str, Any]:
    resolved = resolve_contract(
        method=args.method,
        seed=args.seed,
        world_size=args.world_size,
        max_updates=args.max_updates,
    )
    assets = validate_assets(
        _asset_arguments(args),
        purpose="train",
        require_existing=require_existing,
    )
    validate_rlinf_checkout(args.rlinf_root, require_existing=require_existing)
    if require_existing and args.resume is not None and not args.resume.is_file():
        raise ContractError(f"resume checkpoint is not a file: {args.resume}")
    return {
        "schema": SCHEMA,
        "status": "DRY_RUN" if not require_existing else "PREFLIGHT_PASS",
        "operation": "train",
        "executes_benchmark": require_existing,
        "paper_contract": paper_contract_dict(),
        "resolved": resolved.as_dict(),
        "assets": {
            name: asset_fingerprint(path) if require_existing and path is not None else str(path)
            for name, path in assets.items()
        },
        "rlinf_root": str(args.rlinf_root),
        "output_dir": str(args.output_dir),
        "resume": str(args.resume) if args.resume is not None else None,
        "runtime": {
            "launcher": (
                "torchrun --standalone --nproc-per-node "
                f"{args.world_size} benchmarks/maniskill/train.py ..."
            ),
            "imports_deferred_until_real_run": [
                "torch",
                "omegaconf",
                "rlinf",
                "openpi",
                "mani_skill",
            ],
            "dry_run_performs_training": False,
        },
    }


def _tree_to_cpu(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_cpu(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item, torch) for item in value)
    return value


def _tree_detach_clone(value: Any, torch: Any) -> Any:
    """Convert rollout outputs to ordinary tensors before autograd re-use.

    ManiSkill action sampling runs under ``torch.inference_mode`` for memory
    and speed.  The same rollout context is then consumed by NFT/SFT calls,
    which must run with autograd enabled.  Cloning every tensor here makes
    that boundary explicit and avoids passing inference tensors into a
    differentiable model call.
    """

    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _tree_detach_clone(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_detach_clone(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_detach_clone(item, torch) for item in value)
    return value


def _tree_to_device(value: Any, device: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_device(item, device, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device, torch) for item in value)
    return value


def _capture_rng(torch: Any, np: Any, device: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }


def _restore_rng(state: Mapping[str, Any], torch: Any, np: Any, device: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)


def _seed_all(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model_config(
    OmegaConf: Any,
    *,
    model_path: Path,
    normalization_stats: Path,
    train_expert_only: bool,
) -> Any:
    return OmegaConf.create(
        {
            "model_path": str(model_path),
            "openpi_data": {"norm_stats_path": str(normalization_stats)},
            "openpi": {
                "config_name": "pi05_maniskill",
                "num_images_in_input": 1,
                "noise_method": "flow_sde",
                "joint_logprob": False,
                "action_horizon": CONTRACT.model_horizon,
                "action_chunk": CONTRACT.executed_prefix,
                "action_env_dim": CONTRACT.physical_action_dims,
                "num_steps": CONTRACT.solver_steps,
                "train_expert_only": train_expert_only,
                "add_value_head": False,
            },
        }
    )


def _environment_config(OmegaConf: Any) -> Any:
    return OmegaConf.create(
        {
            "env_type": "maniskill",
            "total_num_envs": 1,
            "auto_reset": False,
            "ignore_terminations": True,
            "use_rel_reward": True,
            "use_full_state": False,
            "reward_mode": "raw",
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": False,
            "max_steps_per_rollout_epoch": CONTRACT.episode_primitive_steps,
            "max_episode_steps": CONTRACT.episode_primitive_steps,
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "",
            },
            "enable_offload": False,
            "init_params": {
                "id": CONTRACT.task,
                "num_envs": 1,
                "obs_mode": "rgb+segmentation",
                "control_mode": MANISKILL_CONTROL_MODE,
                "sim_backend": "gpu",
                "sim_config": {
                    "sim_freq": CONTRACT.simulation_frequency_hz,
                    "control_freq": CONTRACT.control_frequency_hz,
                },
                "max_episode_steps": CONTRACT.episode_primitive_steps,
                "sensor_configs": {"shader_pack": "default"},
                "render_mode": "all",
                "obj_set": "train",
                "use_multiple_plates": False,
            },
        }
    )


def _fixed_noise(torch: Any, device: Any, batch_size: int = 1) -> Any:
    noise = torch.linspace(
        -1.0,
        1.0,
        steps=CONTRACT.model_horizon * 32,
        dtype=torch.float32,
        device=device,
    ).reshape(1, CONTRACT.model_horizon, 32)
    return noise.expand(batch_size, -1, -1).contiguous()


def _minimal_context(forward_inputs: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    missing = [key for key in CONTEXT_KEYS if key not in forward_inputs]
    if missing:
        raise RuntimeError(f"OpenPI rollout is missing NFT context keys: {missing}")
    context: dict[str, Any] = {}
    for key in CONTEXT_KEYS:
        value = forward_inputs[key]
        if not torch.is_tensor(value):
            raise TypeError(f"OpenPI NFT context {key!r} is not a tensor")
        context[key] = value.repeat_interleave(CONTRACT.solver_steps, dim=0).detach().clone()
    return context


def _teacher_versions(teacher: Any) -> tuple[tuple[str, int], ...]:
    return tuple((name, parameter._version) for name, parameter in teacher.named_parameters())


def _assert_teacher_unchanged(teacher: Any, initial_versions: tuple[tuple[str, int], ...]) -> None:
    observed = _teacher_versions(teacher)
    if observed != initial_versions:
        raise RuntimeError("frozen teacher parameter version changed")
    if teacher.training:
        raise RuntimeError("frozen teacher left evaluation mode")
    if any(
        parameter.requires_grad or parameter.grad is not None for parameter in teacher.parameters()
    ):
        raise RuntimeError("frozen teacher acquired trainability or gradients")


def _allreduce_gradients(
    named_parameters: list[tuple[str, Any]],
    torch: Any,
    world_size: int,
) -> None:
    missing = [name for name, parameter in named_parameters if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in named_parameters
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if missing or nonfinite:
        raise RuntimeError(f"gradient contract failed: missing={missing}, nonfinite={nonfinite}")
    if world_size > 1:
        for _name, parameter in named_parameters:
            torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _gather_object(value: Any, torch: Any, rank: int, world_size: int) -> list[Any] | None:
    if world_size == 1:
        return [value]
    output = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object(value, output, dst=0)
    return output


def _checkpoint_state_to_cpu(value: Any, torch: Any) -> Any:
    return _tree_to_cpu(value, torch)


def _release_asset_hashes(args: argparse.Namespace) -> dict[str, str]:
    """Hash every caller-owned input that changes the meaning of a resume."""

    return {
        name: tree_sha256(path) for name, path in _asset_arguments(args).items() if path is not None
    }


def _resume_asset_hashes(
    args: argparse.Namespace,
    release_assets: Mapping[str, str],
    release_manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Select resume inputs while accepting pre-v2 manifests safely.

    Old checkpoints have no teacher-normalizer entry.  They are compatible only
    when the caller did not request an override (the teacher then inherits the
    student path, exactly as in the old worker).  An explicit override must be
    present in the checkpoint identity and is rejected otherwise.
    """

    recorded = release_manifest.get("asset_sha256")
    override = getattr(args, "teacher_normalization_stats", None)
    if (
        override is None
        and isinstance(recorded, Mapping)
        and "teacher_normalization_stats" not in recorded
    ):
        return {
            name: value
            for name, value in release_assets.items()
            if name != "teacher_normalization_stats"
        }
    return dict(release_assets)


def _release_code_provenance(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "release_source_sha256": tree_sha256(
            Path(__file__).resolve().parents[2],
            include=("src", "benchmarks", "configs"),
        ),
        "rlinf_revision": git_revision(args.rlinf_root),
    }


def _release_checkpoint_identity(
    args: argparse.Namespace,
    resolved: Any,
    *,
    world_size: int,
    update: int,
    assets: Mapping[str, str] | None = None,
    code: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    return build_checkpoint_manifest(
        benchmark="maniskill",
        method=args.method,
        seed=args.seed,
        world_size=world_size,
        update=update,
        contract=paper_contract_dict(),
        assets=dict(assets or _release_asset_hashes(args)),
        code=dict(code or _release_code_provenance(args)),
        extra={"resolved_contract": resolved.as_dict()},
    )


def _save_checkpoint(
    *,
    args: argparse.Namespace,
    resolved: Any,
    torch: Any,
    np: Any,
    rank: int,
    world_size: int,
    device: Any,
    student: Any,
    optimizer: Any,
    env: Any,
    observation: Any,
    update: int,
    reset_count: int,
    release_assets: Mapping[str, str],
    release_code: Mapping[str, str | None],
) -> Path | None:
    rank_state = {
        "rank": rank,
        "rng": _capture_rng(torch, np, device),
        "environment": env.get_state(),
        "observation": _tree_to_cpu(observation, torch),
        "reset_count": reset_count,
    }
    rank_states = _gather_object(rank_state, torch, rank, world_size)
    if rank != 0:
        if world_size > 1:
            torch.distributed.barrier()
        return None
    chunks = update * world_size
    directory = args.output_dir / "checkpoints" / f"chunks_{chunks:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    temporary = directory / f".checkpoint.{os.getpid()}.tmp"
    release_manifest = _release_checkpoint_identity(
        args,
        resolved,
        world_size=world_size,
        update=update,
        assets=release_assets,
        code=release_code,
    )
    payload = {
        "schema": SCHEMA,
        "model": _checkpoint_state_to_cpu(student.state_dict(), torch),
        "optimizer": _checkpoint_state_to_cpu(optimizer.state_dict(), torch),
        "cursor": update,
        "rank_states": rank_states,
        "manifest": {
            "method": args.method,
            "seed": args.seed,
            "world_size": world_size,
            "optimizer_updates": update,
            "fresh_chunks": chunks,
            "contract": resolved.as_dict(),
            "release_contract": release_manifest,
        },
    }
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)
    digest = sha256_file(checkpoint)
    (directory / "checkpoint.sha256").write_text(f"{digest}  checkpoint.pt\n", encoding="ascii")
    write_json(
        directory / "manifest.json",
        {
            **payload["manifest"],
            "release_contract": release_manifest,
            "checkpoint_sha256": digest,
            "distributed_rank_states": len(rank_states or []),
        },
    )
    if world_size > 1:
        torch.distributed.barrier()
    return checkpoint


def _load_resume(
    *,
    args: argparse.Namespace,
    torch: Any,
    np: Any,
    rank: int,
    world_size: int,
    device: Any,
    student: Any,
    optimizer: Any,
    env: Any,
    release_assets: Mapping[str, str],
    release_code: Mapping[str, str | None],
) -> tuple[int, Any, int]:
    if args.resume is None:
        observation, _ = env.reset(seed=args.seed + rank, options={})
        return 0, _tree_to_device(observation, device, torch), 0
    payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise RuntimeError("resume checkpoint schema mismatch")
    manifest = payload.get("manifest", {})
    expected = {
        "method": args.method,
        "seed": args.seed,
        "world_size": world_size,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"resume {key} mismatch: {manifest.get(key)!r} != {value!r}")
    release_manifest = manifest.get("release_contract")
    if not isinstance(release_manifest, Mapping):
        raise RuntimeError("resume checkpoint lacks release contract manifest")
    try:
        validate_checkpoint_manifest(
            release_manifest,
            expected={
                "benchmark": "maniskill",
                "method": args.method,
                "seed": args.seed,
                "world_size": world_size,
                "contract": paper_contract_dict(),
            },
            assets=_resume_asset_hashes(args, release_assets, release_manifest),
            code=release_code,
        )
    except ReleaseContractError as error:
        raise RuntimeError(f"resume release contract mismatch: {error}") from error
    student.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)
    rank_states = payload.get("rank_states")
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise RuntimeError("resume checkpoint lacks one exact state per rank")
    state = rank_states[rank]
    if state.get("rank") != rank:
        raise RuntimeError("resume rank-state ordering mismatch")
    env.load_state(state["environment"])
    observation = _tree_to_device(state["observation"], device, torch)
    _restore_rng(state["rng"], torch, np, device)
    return int(payload["cursor"]), observation, int(state["reset_count"])


def _runtime(args: argparse.Namespace, preflight: Mapping[str, Any]) -> dict[str, Any]:
    # External code is imported only after explicit paths passed preflight.
    args.rlinf_root = args.rlinf_root.expanduser().resolve()
    host_integration = validate_rlinf_host_support(
        args.rlinf_root,
        benchmark="maniskill",
        patch_path=Path(__file__).resolve().parents[2]
        / "integrations"
        / "rlinf"
        / "path-opd-host-support.patch",
        require_openpi_distribution=True,
    )
    sys.path.insert(0, str(args.rlinf_root))
    os.environ["MANISKILL_ASSET_DIR"] = str(args.simulator_assets.expanduser().resolve())
    os.environ["MS_ASSET_DIR"] = str(args.maniskill_package_assets.expanduser().resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.action_utils import prepare_actions
    from rlinf.envs.maniskill.maniskill_offload_env import _ManiskillEnvCore
    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.models.embodiment.openpi import get_model

    from path_opd.adapters.openpi import OpenPIAdapter
    from path_opd.core import (
        ActionContract,
        FlowSchedule,
        FrozenTeacher,
        PathOPD,
        configure_action_expert,
        validate_exact_trace,
    )

    launched_world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if launched_world != args.world_size:
        raise RuntimeError(
            f"launcher WORLD_SIZE={launched_world} does not match --world-size={args.world_size}"
        )
    if args.world_size > 1 and "RANK" not in os.environ:
        raise RuntimeError("multi-rank execution must be launched with torchrun")
    if not torch.cuda.is_available():
        raise RuntimeError("real ManiSkill training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.world_size > 1:
        torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))

    resolved = resolve_contract(
        method=args.method,
        seed=args.seed,
        world_size=args.world_size,
        max_updates=args.max_updates,
    )
    _seed_all(args.seed + rank, torch, np)
    release_assets = _release_asset_hashes(args)
    release_code = _release_code_provenance(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics" / f"rank_{rank:04d}.jsonl"
    env = None
    started = time.time()
    try:
        student = get_model(
            _model_config(
                OmegaConf,
                model_path=args.base_model,
                normalization_stats=args.normalization_stats,
                train_expert_only=True,
            )
        ).to(device)
        trainable_names = configure_action_expert(student)
        named_trainable = sorted(
            (name, parameter)
            for name, parameter in student.named_parameters()
            if parameter.requires_grad
        )
        if tuple(name for name, _ in named_trainable) != tuple(sorted(trainable_names)):
            raise RuntimeError("action-expert selector and requires_grad topology differ")
        teacher = get_model(
            _model_config(
                OmegaConf,
                model_path=args.teacher_model,
                normalization_stats=teacher_normalization_stats(args),
                train_expert_only=False,
            )
        ).to(device)
        teacher.eval().requires_grad_(False)
        teacher_versions = _teacher_versions(teacher)
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named_trainable],
            lr=CONTRACT.learning_rate,
            betas=CONTRACT.betas,
            eps=CONTRACT.epsilon,
            weight_decay=CONTRACT.weight_decay,
        )
        objective = PathOPD(
            ActionContract(CONTRACT.executed_prefix, CONTRACT.physical_action_dims),
            FlowSchedule.uniform(CONTRACT.solver_steps),
        )
        path_teacher_inputs: Mapping[str, Any] | None = None

        def teacher_velocity(model: Any, states: Any, times: Any) -> Any:
            if path_teacher_inputs is None:
                raise RuntimeError("Path-OPD teacher context was not bound")
            return OpenPIAdapter(
                model=model,
                forward_inputs=path_teacher_inputs,
                steps=CONTRACT.solver_steps,
                nft_forward_type=ForwardType.NFT,
                contract=ActionContract(CONTRACT.executed_prefix, CONTRACT.physical_action_dims),
            ).velocity(states, times)

        # A multi-billion-parameter teacher is guarded by tensor identity,
        # metadata, and PyTorch version counters during training.  Byte-level
        # hashing remains available as an explicit offline audit, but copying
        # every weight to CPU here would make startup needlessly expensive.
        frozen_teacher = FrozenTeacher(
            teacher, teacher_velocity, deep_fingerprint=False
        )
        env = _ManiskillEnvCore(
            cfg=_environment_config(OmegaConf),
            num_envs=1,
            seed_offset=args.seed + rank,
            # RLinf uses this value when deriving process-global seeds and
            # worker identities. Each rank owns one local environment, but
            # the process count is still the distributed world size; using 1
            # here would make every rank look like the sole process.
            total_num_processes=args.world_size,
            worker_info=None,
            record_metrics=False,
        )
        # Training never consumes success/outcome fields; record_metrics=False
        # alone is insufficient in this historical RLinf wrapper.
        env._record_metrics = lambda _reward, infos: infos
        update, observation, reset_count = _load_resume(
            args=args,
            torch=torch,
            np=np,
            rank=rank,
            world_size=args.world_size,
            device=device,
            student=student,
            optimizer=optimizer,
            env=env,
            release_assets=release_assets,
            release_code=release_code,
        )
        if update > resolved.max_updates:
            raise RuntimeError("resume cursor exceeds --max-updates")
        saved_chunks: set[int] = set()
        last_loss: float | None = None
        while update < resolved.max_updates:
            next_update = update + 1
            student.eval()
            initial_noise = _fixed_noise(torch, device)
            with torch.inference_mode():
                raw_actions, rollout = student.predict_action_batch(
                    observation,
                    mode="eval",
                    compute_values=False,
                    initial_noise=initial_noise,
                )
            forward_inputs = rollout.get("forward_inputs")
            if not isinstance(forward_inputs, Mapping):
                raise RuntimeError("OpenPI rollout did not return forward_inputs")
            # The rollout was sampled under inference_mode; detach and clone
            # the complete context before any NFT/DAgger call can backprop.
            forward_inputs = _tree_detach_clone(dict(forward_inputs), torch)
            rollout = dict(rollout)
            rollout["forward_inputs"] = forward_inputs
            chains = forward_inputs.get("chains")
            model_actions = rollout.get("model_actions")
            if not torch.is_tensor(chains) or not torch.is_tensor(model_actions):
                raise RuntimeError("OpenPI rollout did not return tensor chains/endpoints")
            provenance = validate_exact_trace(
                chains,
                model_actions,
                raw_actions,
                forward_inputs["action"],
                expected_steps=CONTRACT.solver_steps,
            )
            if tuple(chains.shape[1:]) != (
                CONTRACT.solver_steps + 1,
                CONTRACT.model_horizon,
                32,
            ):
                raise RuntimeError(f"unexpected OpenPI chain shape: {tuple(chains.shape)}")
            student_adapter = OpenPIAdapter.from_rollout(
                student,
                rollout,
                steps=CONTRACT.solver_steps,
                nft_forward_type=ForwardType.NFT,
                sft_forward_type=ForwardType.SFT,
                contract=ActionContract(CONTRACT.executed_prefix, CONTRACT.physical_action_dims),
            )
            submitted = prepare_actions(
                raw_chunk_actions=raw_actions,
                env_type="maniskill",
                model_type="openpi",
                num_action_chunks=CONTRACT.executed_prefix,
                action_dim=CONTRACT.physical_action_dims,
                policy="widowx_bridge",
            )
            expected_action_shape = (
                1,
                CONTRACT.executed_prefix,
                CONTRACT.physical_action_dims,
            )
            if tuple(submitted.shape) != expected_action_shape:
                raise RuntimeError(f"unexpected environment action shape: {tuple(submitted.shape)}")
            observations, _rewards, _terminations, _truncations, _infos = env.chunk_step(submitted)
            next_observation = _tree_to_device(observations[-1], device, torch)

            optimizer.zero_grad(set_to_none=True)
            student.train()
            if args.method == "path_opd":
                path_teacher_inputs = forward_inputs
                result = objective.supervise(
                    student_adapter.chains,
                    student_adapter.velocity,
                    frozen_teacher,
                )
                loss = result.loss
                per_time = [float(value) for value in result.per_time.detach().cpu()]
                path_states_detached = not result.trace.states.requires_grad
            else:
                with torch.no_grad():
                    _teacher_actions, teacher_rollout = teacher.predict_action_batch(
                        observation,
                        mode="eval",
                        compute_values=False,
                        initial_noise=initial_noise,
                    )
                teacher_target = teacher_rollout["model_actions"].detach()
                _seed_all(args.seed + rank + next_update * args.world_size, torch, np)
                loss = student_adapter.endpoint_dagger_loss(teacher_target)
                per_time = None
                path_states_detached = None
            if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss):
                raise RuntimeError("training loss is not one finite scalar tensor")
            loss.backward()
            _allreduce_gradients(named_trainable, torch, args.world_size)
            total_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named_trainable],
                CONTRACT.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if args.world_size > 1:
                torch.distributed.barrier()
            _assert_teacher_unchanged(teacher, teacher_versions)
            update = next_update
            last_loss = float(loss.detach().cpu())
            observation = next_observation
            if update % 16 == 0:
                reset_count += 1
                observation, _ = env.reset(
                    seed=(
                        args.seed + update * args.world_size + rank + reset_count * args.world_size
                    ),
                    options={},
                )
                observation = _tree_to_device(observation, device, torch)
            global_loss = torch.tensor(last_loss, device=device, dtype=torch.float64)
            if args.world_size > 1:
                torch.distributed.all_reduce(global_loss)
                global_loss.div_(args.world_size)
            _append_jsonl(
                metrics_path,
                {
                    "schema": SCHEMA,
                    "rank": rank,
                    "method": args.method,
                    "seed": args.seed,
                    "optimizer_updates": update,
                    "fresh_chunks": update * args.world_size,
                    "loss": last_loss,
                    "global_loss_mean": float(global_loss.cpu()),
                    "per_time_loss": per_time,
                    "path_states_detached": path_states_detached,
                    "gradient_norm_before_clip": float(total_norm.detach().cpu()),
                    "trace_provenance": provenance,
                    "success_fields_read": False,
                },
            )
            chunks = update * args.world_size
            should_save = chunks in CONTRACT.checkpoint_chunks or update == resolved.max_updates
            if should_save and chunks not in saved_chunks:
                _save_checkpoint(
                    args=args,
                    resolved=resolved,
                    torch=torch,
                    np=np,
                    rank=rank,
                    world_size=args.world_size,
                    device=device,
                    student=student,
                    optimizer=optimizer,
                    env=env,
                    observation=observation,
                    update=update,
                    reset_count=reset_count,
                    release_assets=release_assets,
                    release_code=release_code,
                )
                saved_chunks.add(chunks)
        _assert_teacher_unchanged(teacher, teacher_versions)
        summary = {
            "schema": SCHEMA,
            "status": "PASS",
            "operation": "train",
            "method": args.method,
            "training_seed": args.seed,
            "rank": rank,
            "world_size": args.world_size,
            "optimizer_updates": update,
            "fresh_chunks": update * args.world_size,
            "formal_full_budget": resolved.formal_full_budget,
            "qualification": resolved.qualification,
            "last_loss": last_loss,
            "trainable_parameter_tensors": len(named_trainable),
            "teacher_frozen": True,
            "success_fields_read": False,
            "elapsed_seconds": time.time() - started,
            "preflight": {**preflight, "rlinf_host_integration": host_integration},
        }
        write_json(args.output_dir / "summaries" / f"rank_{rank:04d}.json", summary)
        if args.world_size > 1:
            torch.distributed.barrier()
        return summary
    finally:
        if env is not None:
            env.close()
        if args.world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _synthetic_smoke(argv: list[str]) -> int:
    """Run the dependency-light algorithm proof through this public entrypoint."""

    parser = argparse.ArgumentParser(
        prog="maniskill-train --synthetic-smoke",
        description=(
            "Run the synthetic ManiSkill runner contract; no RLinf checkout, "
            "model, simulator, CUDA device, or benchmark asset is used."
        ),
    )
    parser.add_argument("--synthetic-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("artifacts/runner-smoke/maniskill")
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
        "maniskill",
        args.work_dir,
        entrypoint="benchmarks/maniskill/train.py",
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
            print(f"maniskill-train synthetic smoke: {error}", file=sys.stderr)
            return 2
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    try:
        report = _resolved_report(args, require_existing=not args.dry_run)
        if args.dry_run:
            write_json(args.contract_output, report)
            return 0
        _runtime(args, report)
        return 0
    except (ContractError, ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"maniskill-train: {error}", file=sys.stderr)
        if os.environ.get("PATH_OPD_DEBUG") == "1":
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

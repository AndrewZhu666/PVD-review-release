#!/usr/bin/env python3

# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Evaluate one OpenPI checkpoint on the fixed 320-row ManiSkill panel.

This is a portable extraction of the Apache-2.0 RLinf fixed-panel evaluator.
It imports and runs the real RLinf OpenPI policy, action adapter, and ManiSkill
environment.  Internal cluster scheduling, machine ownership probes, private
paths, and private digest constants are excluded; checkpoint and panel hashes
must instead be supplied explicitly by the reviewer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

from path_opd.release_contract import (
    ReleaseContractError,
    formal_panel_contract,
    git_revision,
    tree_sha256,
    validate_rlinf_host_support,
    verify_panel_digest,
)
from path_opd.release_contract import (
    validate_checkpoint_manifest as validate_release_checkpoint_manifest,
)

if __package__:
    from .common import (
        CONTRACT,
        MANISKILL_CONTROL_MODE,
        ContractError,
        asset_fingerprint,
        paper_contract_dict,
        rows_for_rank,
        sha256_file,
        validate_assets,
        validate_panel,
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
        rows_for_rank,
        sha256_file,
        validate_assets,
        validate_panel,
        validate_rlinf_checkout,
        write_json,
    )


SCHEMA = "path-opd-maniskill-evaluation-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

RESULT_SCHEMA = {
    "schema": SCHEMA,
    "status": "PASS",
    "benchmark": "maniskill",
    "protocol": {
        "name": CONTRACT.evaluation_protocol,
        "solver_steps": CONTRACT.solver_steps,
        "rows": CONTRACT.evaluation_rows,
        "episode_primitive_steps": CONTRACT.episode_primitive_steps,
    },
    "coverage": {
        "expected_rows": "integer",
        "actual_rows": "integer",
        "unique_reset_ids": "integer",
        "missing_rows": "integer",
        "duplicate_rows": "integer",
    },
    "metrics": {
        "success_once": {"count": "integer", "denominator": "integer", "rate": "float"},
        "success_at_end": {"count": "integer", "denominator": "integer", "rate": "float"},
    },
    "rows": [
        {
            "panel_index": "integer",
            "reset_episode_id": "integer",
            "worker_rank": "integer",
            "env_slot": "integer",
            "success_once": "boolean",
            "success_at_end": "boolean",
            "episode_len": "integer",
        }
    ],
}


def _sha256_argument(value: str) -> str:
    normalized = value.lower()
    if _SHA256.fullmatch(normalized) is None:
        raise argparse.ArgumentTypeError("expected 64 hexadecimal SHA-256 digits")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RLinf/OpenPI fixed 320-row ManiSkill evaluator",
    )
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", type=_sha256_argument, required=True)
    parser.add_argument(
        "--method",
        choices=CONTRACT.method_choices,
        required=True,
        help="training objective recorded by the checkpoint manifest",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        choices=CONTRACT.supported_seeds,
        required=True,
        help="training seed recorded by the checkpoint manifest",
    )
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-sha256", type=_sha256_argument, required=True)
    parser.add_argument(
        "--panel-mode",
        choices=("formal", "custom"),
        default="formal",
        help=(
            "bind the panel to the published formal digest, or explicitly mark "
            "a caller-owned panel as custom"
        ),
    )
    parser.add_argument("--world-size", type=int, default=CONTRACT.world_size)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--contract-output",
        type=Path,
        help="write dry-run JSON here instead of standard output",
    )
    return parser


def _asset_arguments(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "base_model": args.base_model,
        "normalization_stats": args.normalization_stats,
        "simulator_assets": args.simulator_assets,
        "maniskill_package_assets": args.maniskill_package_assets,
        "evaluation_panel": args.panel,
    }


def _checkpoint_manifest_path(checkpoint: Path) -> Path:
    """Return the sidecar emitted next to a ManiSkill training checkpoint."""

    return checkpoint.with_name("manifest.json")


def _load_checkpoint_manifest(path: Path) -> dict[str, Any]:
    manifest_path = _checkpoint_manifest_path(path)
    if not manifest_path.is_file():
        raise ContractError(
            "checkpoint manifest.json is required next to checkpoint.pt: "
            f"{manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(
            f"could not read checkpoint manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ContractError("checkpoint manifest must be a JSON object")
    return payload


def _checkpoint_asset_hashes(args: argparse.Namespace) -> dict[str, str]:
    """Hash the evaluator-visible assets used to validate release provenance."""

    return {
        name: tree_sha256(path)
        for name, path in {
            "base_model": args.base_model,
            "normalization_stats": args.normalization_stats,
            "simulator_assets": args.simulator_assets,
            "maniskill_package_assets": args.maniskill_package_assets,
        }.items()
    }


def _validate_checkpoint_contract(
    args: argparse.Namespace,
    *,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Validate a training manifest before importing CUDA or RLinf runtime code."""

    manifest = _load_checkpoint_manifest(args.checkpoint)
    if manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise ContractError(
            "checkpoint manifest checkpoint_sha256 does not match the supplied checkpoint"
        )
    expected_top_level = {
        "method": args.method,
        "seed": args.training_seed,
        "world_size": CONTRACT.world_size,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected_top_level.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ContractError(f"checkpoint manifest does not match evaluation: {mismatches}")
    release = manifest.get("release_contract")
    if not isinstance(release, Mapping):
        raise ContractError("checkpoint manifest lacks release_contract provenance")
    recorded_assets = release.get("asset_sha256")
    required_assets = {
        "base_model",
        "frozen_teacher",
        "normalization_stats",
        "simulator_assets",
        "maniskill_package_assets",
    }
    if not isinstance(recorded_assets, Mapping):
        raise ContractError("checkpoint release contract lacks asset_sha256")
    missing_assets = sorted(required_assets - set(recorded_assets))
    if missing_assets:
        raise ContractError(
            "checkpoint release contract lacks asset(s): " + ", ".join(missing_assets)
        )
    malformed_assets = [
        name
        for name in required_assets
        if not isinstance(recorded_assets.get(name), str)
        or _SHA256.fullmatch(recorded_assets[name].lower()) is None
    ]
    if malformed_assets:
        raise ContractError(
            "checkpoint release contract has invalid asset SHA-256 value(s): "
            + ", ".join(sorted(malformed_assets))
        )
    asset_hashes = _checkpoint_asset_hashes(args)
    code_hashes = {
        "release_source_sha256": tree_sha256(
            Path(__file__).resolve().parents[2], include=("src", "benchmarks", "configs")
        ),
        "rlinf_revision": git_revision(args.rlinf_root),
    }
    try:
        validate_release_checkpoint_manifest(
            release,
            expected={
                "benchmark": CONTRACT.benchmark,
                "method": args.method,
                "seed": args.training_seed,
                "world_size": CONTRACT.world_size,
                "contract": paper_contract_dict(),
            },
            assets=asset_hashes,
            code=code_hashes,
        )
    except ReleaseContractError as error:
        raise ContractError(f"checkpoint release contract mismatch: {error}") from error

    nested_contract = manifest.get("contract")
    if not isinstance(nested_contract, Mapping):
        raise ContractError("checkpoint manifest lacks resolved training contract")
    expected_contract = {
        "method": args.method,
        "seed": args.training_seed,
        "world_size": CONTRACT.world_size,
        "solver_steps": CONTRACT.solver_steps,
        "model_horizon": CONTRACT.model_horizon,
        "executed_prefix": CONTRACT.executed_prefix,
        "physical_action_dims": CONTRACT.physical_action_dims,
        "global_batch_size": CONTRACT.world_size,
    }
    contract_mismatches = {
        key: (nested_contract.get(key), value)
        for key, value in expected_contract.items()
        if nested_contract.get(key) != value
    }
    if contract_mismatches:
        raise ContractError(
            "checkpoint resolved contract does not match evaluation: "
            f"{contract_mismatches}"
        )
    args._checkpoint_manifest = manifest
    args._checkpoint_release_contract = dict(release)
    return {
        "verified": True,
        "method": args.method,
        "training_seed": args.training_seed,
        "world_size": CONTRACT.world_size,
        "release_contract_schema": release.get("schema"),
        "asset_hashes_verified": sorted(asset_hashes),
        "unverifiable_assets": ["frozen_teacher"],
        "code_hashes_verified": sorted(code_hashes),
    }


def _preflight(args: argparse.Namespace, *, require_existing: bool) -> dict[str, Any]:
    if args.world_size != CONTRACT.world_size:
        raise ContractError(
            f"the recorded panel ownership requires world_size={CONTRACT.world_size}"
        )
    if args.policy_seed < 0:
        raise ContractError("policy_seed must be non-negative")
    if require_existing:
        if not args.checkpoint.is_file():
            raise ContractError(f"checkpoint is not a file: {args.checkpoint}")
        # Check the sidecar before unrelated model directories so a bare or
        # legacy checkpoint fails with the actionable provenance error.
        _load_checkpoint_manifest(args.checkpoint)
    assets = validate_assets(
        _asset_arguments(args),
        purpose="evaluate",
        require_existing=require_existing,
    )
    panel_summary: dict[str, Any] | None = None
    panel_contract = formal_panel_contract("maniskill")
    if args.panel_mode == "formal" and not panel_contract.has_formal_digest:
        raise ContractError(
            "ManiSkill has no published formal panel digest; use --panel-mode custom"
        )
    if require_existing:
        validate_rlinf_checkout(args.rlinf_root, require_existing=True)
        checkpoint_sha = sha256_file(args.checkpoint)
        if checkpoint_sha != args.checkpoint_sha256:
            raise ContractError(
                f"checkpoint SHA-256 mismatch: {checkpoint_sha} != {args.checkpoint_sha256}"
            )
        checkpoint_contract = _validate_checkpoint_contract(
            args, checkpoint_sha256=checkpoint_sha
        )
        panel = validate_panel(args.panel)
        if panel["file_sha256"] != args.panel_sha256:
            raise ContractError(
                f"panel SHA-256 mismatch: {panel['file_sha256']} != {args.panel_sha256}"
            )
        try:
            panel_verification = verify_panel_digest(
                "maniskill",
                panel["file_sha256"],
                mode=args.panel_mode,
                supplied_sha256=args.panel_sha256,
                rows=panel["row_count"],
            )
        except ReleaseContractError as error:
            raise ContractError(str(error)) from error
        panel_summary = {
            "rows": panel["row_count"],
            "file_sha256": panel["file_sha256"],
            "content_sha256": panel["content_sha256"],
            "ordered_reset_ids_sha256": panel["ordered_reset_ids_sha256"],
            "verification": panel_verification.as_dict(),
        }
    else:
        checkpoint_contract = {
            "verified": False,
            "verification_note": "dry-run only; checkpoint manifest was not required",
        }
    return {
        "schema": SCHEMA,
        "status": "DRY_RUN" if not require_existing else "PREFLIGHT_PASS",
        "operation": "evaluate",
        "executes_benchmark": require_existing,
        "paper_contract": paper_contract_dict(),
        "resolved": {
            "world_size": args.world_size,
            "policy_seed": args.policy_seed,
            "solver_steps": CONTRACT.solver_steps,
            "model_horizon": CONTRACT.model_horizon,
            "executed_prefix": CONTRACT.executed_prefix,
            "physical_action_dims": CONTRACT.physical_action_dims,
            "fixed_rows": CONTRACT.evaluation_rows,
            "episode_primitive_steps": CONTRACT.episode_primitive_steps,
            "inference_only": True,
        },
        "assets": {
            name: asset_fingerprint(path) if require_existing and path is not None else str(path)
            for name, path in assets.items()
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "expected_sha256": args.checkpoint_sha256,
            "contract": checkpoint_contract,
        },
        "panel": {
            "path": str(args.panel),
            "expected_sha256": args.panel_sha256,
            "mode": args.panel_mode,
            "formal_contract": panel_contract.as_dict(),
            "validated": panel_summary,
        },
        "rlinf_root": str(args.rlinf_root),
        "output": str(args.output),
        "output_schema": RESULT_SCHEMA,
        "runtime": {
            "launcher": (
                "torchrun --standalone --nproc-per-node 8 benchmarks/maniskill/evaluate.py ..."
            ),
            "imports_deferred_until_real_run": [
                "torch",
                "omegaconf",
                "rlinf",
                "openpi",
                "mani_skill",
            ],
            "dry_run_performs_evaluation": False,
        },
    }


def _seed_all(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tree_to_device(value: Any, device: Any, torch: Any) -> Any:
    """Move every tensor in an environment observation to the policy device."""

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_device(item, device, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device, torch) for item in value)
    return value


def _model_config(OmegaConf: Any, args: argparse.Namespace) -> Any:
    return OmegaConf.create(
        {
            "model_path": str(args.base_model),
            "openpi_data": {"norm_stats_path": str(args.normalization_stats)},
            "openpi": {
                "config_name": "pi05_maniskill",
                "num_images_in_input": 1,
                "noise_method": "flow_sde",
                "joint_logprob": False,
                "action_horizon": CONTRACT.model_horizon,
                "action_chunk": CONTRACT.executed_prefix,
                "action_env_dim": CONTRACT.physical_action_dims,
                "num_steps": CONTRACT.solver_steps,
                "train_expert_only": False,
                "add_value_head": False,
            },
        }
    )


def _environment_config(OmegaConf: Any, local_rows: int) -> Any:
    return OmegaConf.create(
        {
            "env_type": "maniskill",
            "total_num_envs": CONTRACT.evaluation_rows,
            "auto_reset": True,
            "ignore_terminations": True,
            "use_rel_reward": True,
            "use_full_state": False,
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
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
                "num_envs": local_rows,
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


def _extract_model_state(payload: Any, torch: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        return payload["model"]
    if (
        isinstance(payload, Mapping)
        and payload
        and all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items())
    ):
        return payload
    raise RuntimeError("checkpoint does not contain a model state mapping")


def _tensor_vector(value: Any, name: str, torch: Any, expected: int) -> list[Any]:
    if not torch.is_tensor(value):
        raise RuntimeError(f"evaluation metric {name!r} is not a tensor")
    flat = value.detach().cpu().reshape(-1)
    if flat.numel() != expected:
        raise RuntimeError(
            f"evaluation metric {name!r} has {flat.numel()} rows; expected {expected}"
        )
    return flat.tolist()


def _episode_metrics(
    infos: Mapping[str, Any],
    torch: Any,
    expected: int,
) -> tuple[list[bool], list[bool], list[int]]:
    source = infos.get("final_info", infos)
    if not isinstance(source, Mapping) or not isinstance(source.get("episode"), Mapping):
        raise RuntimeError("final ManiSkill step did not expose episode metrics")
    episode = source["episode"]
    once = [
        bool(value)
        for value in _tensor_vector(episode.get("success_once"), "success_once", torch, expected)
    ]
    at_end = [
        bool(value)
        for value in _tensor_vector(
            episode.get("success_at_end"), "success_at_end", torch, expected
        )
    ]
    lengths = [
        int(value)
        for value in _tensor_vector(episode.get("episode_len"), "episode_len", torch, expected)
    ]
    if lengths != [CONTRACT.episode_primitive_steps] * expected:
        raise RuntimeError("fixed-panel episode length drift")
    return once, at_end, lengths


def _gather_rows(
    rows: list[dict[str, Any]],
    torch: Any,
    rank: int,
    world_size: int,
) -> list[dict[str, Any]] | None:
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object(rows, gathered, dst=0)
    if rank != 0:
        return None
    return [row for shard in gathered or [] for row in shard]


def _runtime(args: argparse.Namespace, preflight: Mapping[str, Any]) -> dict[str, Any] | None:
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
    from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
    from rlinf.models.embodiment.openpi import get_model

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != args.world_size:
        raise RuntimeError(
            f"launcher WORLD_SIZE={world_size} does not match --world-size={args.world_size}"
        )
    if "RANK" not in os.environ:
        raise RuntimeError("fixed-panel evaluation must be launched with torchrun")
    if not torch.cuda.is_available():
        raise RuntimeError("real ManiSkill evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))
    env = None
    started = time.time()
    try:
        panel = validate_panel(args.panel)
        checkpoint_sha = sha256_file(args.checkpoint)
        if checkpoint_sha != args.checkpoint_sha256:
            raise RuntimeError(
                "checkpoint SHA-256 changed after preflight: "
                f"{checkpoint_sha} != {args.checkpoint_sha256}"
            )
        try:
            verify_panel_digest(
                "maniskill",
                panel["file_sha256"],
                mode=args.panel_mode,
                supplied_sha256=args.panel_sha256,
                rows=panel["row_count"],
            )
        except ReleaseContractError as error:
            raise RuntimeError(f"panel contract changed after preflight: {error}") from error
        local_panel_rows = rows_for_rank(panel, rank, world_size)
        local_count = len(local_panel_rows)
        reset_ids = [int(row["reset_episode_id"]) for row in local_panel_rows]
        model = get_model(_model_config(OmegaConf, args)).to(device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("checkpoint payload must be a mapping")
        nested_manifest = checkpoint.get("manifest")
        expected_manifest = getattr(args, "_checkpoint_manifest", None)
        if not isinstance(nested_manifest, Mapping) or not isinstance(expected_manifest, Mapping):
            raise RuntimeError("checkpoint payload lacks the validated manifest")
        for key in ("method", "seed", "world_size", "release_contract"):
            if nested_manifest.get(key) != expected_manifest.get(key):
                raise RuntimeError(f"checkpoint payload {key} disagrees with manifest.json")
        model.load_state_dict(_extract_model_state(checkpoint, torch), strict=True)
        del checkpoint
        model.eval().requires_grad_(False)
        parameter_versions = tuple(
            (name, parameter._version) for name, parameter in model.named_parameters()
        )
        env = ManiskillEnv(
            cfg=_environment_config(OmegaConf, local_count),
            num_envs=local_count,
            seed_offset=rank,
            total_num_processes=world_size,
            worker_info=None,
            record_metrics=True,
        )
        generated = torch.as_tensor(env.reset_state_ids).detach().cpu().tolist()
        if generated != reset_ids:
            raise RuntimeError(
                "pinned ManiSkill seed/topology does not reproduce the supplied panel ownership"
            )
        env.reset_state_ids = torch.tensor(reset_ids, dtype=torch.long, device=env.device)
        observation, _ = env.reset()
        observation = _tree_to_device(observation, device, torch)
        # Reset after model/environment construction so deployment noise is
        # reproducible and independent of initialization implementation details.
        _seed_all(args.policy_seed + rank, torch, np)
        final_infos: Mapping[str, Any] | None = None
        chunks = CONTRACT.episode_primitive_steps // CONTRACT.executed_prefix
        for _chunk_index in range(chunks):
            with torch.inference_mode():
                raw_actions, _rollout = model.predict_action_batch(
                    observation,
                    mode="eval",
                    compute_values=False,
                )
            submitted = prepare_actions(
                raw_chunk_actions=raw_actions,
                env_type="maniskill",
                model_type="openpi",
                num_action_chunks=CONTRACT.executed_prefix,
                action_dim=CONTRACT.physical_action_dims,
                policy="widowx_bridge",
            )
            observations, _rewards, _terminations, truncations, infos = env.chunk_step(submitted)
            observation = _tree_to_device(observations[-1], device, torch)
            final_infos = infos[-1]
        if final_infos is None or not torch.as_tensor(truncations[:, -1]).all():
            raise RuntimeError("fixed-panel evaluation did not terminate all rows at 80 steps")
        success_once, success_at_end, episode_lengths = _episode_metrics(
            final_infos, torch, local_count
        )
        panel_index = {
            int(row["reset_episode_id"]): index for index, row in enumerate(panel["rows"])
        }
        local_results = []
        for slot, (row, once, at_end, episode_len) in enumerate(
            zip(
                local_panel_rows,
                success_once,
                success_at_end,
                episode_lengths,
                strict=True,
            )
        ):
            local_results.append(
                {
                    "panel_index": panel_index[int(row["reset_episode_id"])],
                    "reset_episode_id": int(row["reset_episode_id"]),
                    "worker_rank": rank,
                    "pipeline_stage": int(row.get("pipeline_stage", 0)),
                    "env_slot": int(row.get("env_slot", slot)),
                    "task_id": row.get("task_id"),
                    "task_description": row.get("task_description"),
                    "object_identity": row.get("object_identity"),
                    "success_once": once,
                    "success_at_end": at_end,
                    "episode_len": episode_len,
                }
            )
        observed_versions = tuple(
            (name, parameter._version) for name, parameter in model.named_parameters()
        )
        if observed_versions != parameter_versions:
            raise RuntimeError("evaluation mutated checkpoint parameters")
        if model.training or any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in model.parameters()
        ):
            raise RuntimeError("actor-only evaluation model isolation failed")
        all_rows = _gather_rows(local_results, torch, rank, world_size)
        if rank != 0:
            torch.distributed.barrier()
            return None
        assert all_rows is not None
        all_rows.sort(key=lambda row: row["panel_index"])
        reset_ids_observed = [row["reset_episode_id"] for row in all_rows]
        expected_ids = [int(row["reset_episode_id"]) for row in panel["rows"]]
        if reset_ids_observed != expected_ids:
            raise RuntimeError("gathered evaluation rows do not match panel order and identity")
        denominator = len(all_rows)
        once_count = sum(row["success_once"] for row in all_rows)
        end_count = sum(row["success_at_end"] for row in all_rows)
        result = {
            "schema": SCHEMA,
            "status": "PASS",
            "benchmark": "maniskill",
            "protocol": {
                "name": CONTRACT.evaluation_protocol,
                "solver": "euler",
                "solver_steps": CONTRACT.solver_steps,
                "model_horizon": CONTRACT.model_horizon,
                "executed_prefix": CONTRACT.executed_prefix,
                "physical_action_dims": CONTRACT.physical_action_dims,
                "rows": CONTRACT.evaluation_rows,
                "episode_primitive_steps": CONTRACT.episode_primitive_steps,
                "world_size": world_size,
                "policy_seed_rule": "policy_seed_plus_rank",
                "policy_seed": args.policy_seed,
            },
            "checkpoint": {
                "sha256": args.checkpoint_sha256,
                "model_state_only": True,
                "optimizer_loaded": False,
                "training_authorized": False,
                "contract": preflight.get("checkpoint", {}).get("contract"),
            },
            "panel": {
                "sha256": args.panel_sha256,
                "mode": args.panel_mode,
                "formal": args.panel_mode == "formal",
                "content_sha256": panel["content_sha256"],
                "ordered_reset_ids_sha256": panel["ordered_reset_ids_sha256"],
            },
            "coverage": {
                "expected_rows": CONTRACT.evaluation_rows,
                "actual_rows": denominator,
                "unique_reset_ids": len(set(reset_ids_observed)),
                "missing_rows": CONTRACT.evaluation_rows - denominator,
                "duplicate_rows": denominator - len(set(reset_ids_observed)),
            },
            "metrics": {
                "success_once": {
                    "count": once_count,
                    "denominator": denominator,
                    "rate": once_count / denominator,
                },
                "success_at_end": {
                    "count": end_count,
                    "denominator": denominator,
                    "rate": end_count / denominator,
                },
            },
            "rows": all_rows,
            "model_frozen": True,
            "elapsed_seconds": time.time() - started,
            "preflight": {**preflight, "rlinf_host_integration": host_integration},
        }
        write_json(args.output, result)
        torch.distributed.barrier()
        return result
    finally:
        if env is not None:
            env.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _synthetic_smoke(argv: list[str]) -> int:
    """Run the dependency-light train/checkpoint/panel/trace evaluator proof."""

    parser = argparse.ArgumentParser(
        prog="maniskill-evaluate --synthetic-smoke",
        description=(
            "Run the synthetic ManiSkill evaluator contract; no RLinf checkout, "
            "model, panel, simulator, or CUDA device is used."
        ),
    )
    parser.add_argument("--synthetic-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("artifacts/evaluator-smoke/maniskill")
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
        "maniskill",
        args.work_dir,
        entrypoint="benchmarks/maniskill/evaluate.py",
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
            print(f"maniskill-evaluate synthetic smoke: {error}", file=sys.stderr)
            return 2
    args = build_parser().parse_args(raw_argv)
    try:
        report = _preflight(args, require_existing=not args.dry_run)
        if args.dry_run:
            write_json(args.contract_output, report)
            return 0
        _runtime(args, report)
        return 0
    except (ContractError, ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"maniskill-evaluate: {error}", file=sys.stderr)
        if os.environ.get("PATH_OPD_DEBUG") == "1":
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

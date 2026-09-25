#!/usr/bin/env python3
"""Train the MetaWorld MT50 DAgger or Path-OPD arm.

The runner deliberately keeps RLinf, OpenPI, CUDA, and MetaWorld as runtime
dependencies.  They are imported only after ``--dry-run`` and path validation,
so a reviewer can audit the contract in a CPU-only container.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections.abc import Mapping
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

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

try:
    from .common import (
        PROTOCOL,
        RuntimePaths,
        activate_rlinf,
        atomic_write_json,
        dry_run_plan,
        file_sha256,
        paper_contract,
        print_json,
        validate_task_config,
        validate_train_paths,
    )
except ImportError:  # direct ``python benchmarks/.../train.py`` invocation
    from common import (  # type: ignore[no-redef]
        PROTOCOL,
        RuntimePaths,
        activate_rlinf,
        atomic_write_json,
        dry_run_plan,
        file_sha256,
        paper_contract,
        print_json,
        validate_task_config,
        validate_train_paths,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("path_opd", "endpoint_dagger"),
        required=True,
        help="training objective used for the student update",
    )
    parser.add_argument(
        "--rlinf-checkout",
        type=Path,
        required=True,
        help="caller-selected RLinf checkout (never inferred from the host)",
    )
    parser.add_argument("--base", type=Path, required=True, help="student base model directory")
    parser.add_argument(
        "--teacher", type=Path, required=True, help="frozen teacher model directory"
    )
    parser.add_argument(
        "--norm",
        type=Path,
        required=True,
        help="normalization statistics file used by both models",
    )
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    # The released MT50 protocol has exactly two training seeds.  Keeping the
    # same bound as the evaluator prevents producing a checkpoint that the
    # public evaluation entrypoint must reject later.
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--target-chunks",
        type=int,
        default=PROTOCOL.target_chunks,
        help="fresh global chunks; must be divisible by eight",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        help=(
            "absolute target update count; use 1 for the first qualification run "
            "and a larger value when resuming"
        ),
    )
    parser.add_argument(
        "--qualification",
        action="store_true",
        help="alias for --max-updates 1 on a fresh run",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="optional checkpoint directory containing trainer_state.pt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate paths and print the plan without importing heavy runtime deps",
    )
    return parser


def _validate_resume_target(max_updates: int | None, resumed_updates: object) -> None:
    """Require a resume target to be an absolute update count after the checkpoint."""

    if type(resumed_updates) is not int or resumed_updates < 0:
        raise ValueError("resume manifest has no valid optimizer_update counter")
    if max_updates is not None and max_updates <= resumed_updates:
        raise ValueError(
            "--max-updates is an absolute target; it must be greater than "
            f"the resumed update count ({resumed_updates})"
        )


def _validate_args(args: argparse.Namespace) -> RuntimePaths:
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.target_chunks <= 0 or args.target_chunks % PROTOCOL.world_size:
        raise ValueError("target-chunks must be a positive multiple of 8")
    planned_updates = args.target_chunks // PROTOCOL.world_size
    if args.qualification and args.max_updates is None:
        args.max_updates = 1
    if args.qualification and args.resume is not None:
        raise ValueError(
            "--qualification is only valid for a fresh run; "
            "use --max-updates when resuming"
        )
    if args.max_updates is not None and not 1 <= args.max_updates <= planned_updates:
        raise ValueError("max-updates must be between 1 and target-chunks / 8")
    resume_manifest = None
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
        state_path = args.resume / "trainer_state.pt"
        manifest_path = args.resume / "manifest.json"
        if not args.resume.is_dir() or not state_path.is_file() or not manifest_path.is_file():
            raise ValueError("resume must contain trainer_state.pt and manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid resume manifest: {error}") from error
        expected = {
            "method": args.method,
            "seed": args.seed,
            "world_size": PROTOCOL.world_size,
            "config_name": PROTOCOL.config_name,
            "student_steps": PROTOCOL.student_steps,
            "execution_prefix": PROTOCOL.execution_prefix,
            "physical_action_dims": PROTOCOL.physical_action_dims,
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"resume manifest does not match this run: {mismatches}")
        _validate_resume_target(args.max_updates, manifest.get("optimizer_update"))
        resume_manifest = manifest
    paths = validate_train_paths(
        rlinf_checkout=args.rlinf_checkout,
        base_model=args.base,
        teacher_model=args.teacher,
        norm_stats=args.norm,
        output=args.output,
        require_existing=not args.dry_run,
    )
    if resume_manifest is not None:
        if resume_manifest.get("normalizer_sha256") != file_sha256(paths.norm_stats):
            raise ValueError("resume checkpoint uses different normalization statistics")
        release_assets = _release_asset_hashes(paths)
        release_code = _release_code_provenance(paths)
        release_manifest = resume_manifest.get("release_contract")
        if not isinstance(release_manifest, Mapping):
            raise ValueError("resume manifest lacks release contract provenance")
        try:
            validate_checkpoint_manifest(
                release_manifest,
                expected={
                    "benchmark": PROTOCOL.suite,
                    "method": args.method,
                    "seed": args.seed,
                    "world_size": PROTOCOL.world_size,
                    "contract": _paper_contract(),
                },
                assets=release_assets,
                code=release_code,
            )
        except ReleaseContractError as error:
            raise ValueError(f"resume release contract mismatch: {error}") from error
        args.resume_manifest = resume_manifest
    return paths


def _paper_contract() -> dict[str, Any]:
    return paper_contract()


def _release_asset_hashes(paths: RuntimePaths) -> dict[str, str]:
    if paths.base_model is None or paths.teacher_model is None:
        raise ValueError("training release identity requires base and teacher models")
    return {
        "base_model": tree_sha256(paths.base_model),
        "frozen_teacher": tree_sha256(paths.teacher_model),
        "normalization_stats": tree_sha256(paths.norm_stats),
        "rlinf_task_config": tree_sha256(paths.task_config),
    }


def _release_code_provenance(paths: RuntimePaths) -> dict[str, str | None]:
    return {
        "release_source_sha256": tree_sha256(
            _REPOSITORY_ROOT,
            include=("src", "benchmarks", "configs"),
        ),
        "rlinf_revision": git_revision(paths.rlinf_checkout),
    }


def _release_checkpoint_identity(
    *,
    paths: RuntimePaths,
    args: argparse.Namespace,
    update: int,
    assets: Mapping[str, str] | None = None,
    code: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    return build_checkpoint_manifest(
        benchmark=PROTOCOL.suite,
        method=args.method,
        seed=args.seed,
        world_size=PROTOCOL.world_size,
        update=update,
        contract=_paper_contract(),
        assets=dict(assets or _release_asset_hashes(paths)),
        code=dict(code or _release_code_provenance(paths)),
        extra={"target_chunks": args.target_chunks},
    )


def _seed_all(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _module_sha256(model: Any) -> str:
    import hashlib

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


def _model_config(path: Path, norm: Path, *, train_expert_only: bool) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "model_path": str(path),
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
                "train_expert_only": train_expert_only,
                "add_value_head": False,
                "is_nft": False,
            },
        }
    )


def _load_model(path: Path, norm: Path, *, train_expert_only: bool, device: Any) -> Any:

    from rlinf.models.embodiment.openpi import get_model

    model = get_model(_model_config(path, norm, train_expert_only=train_expert_only))
    model.to(device)
    if not train_expert_only:
        model.requires_grad_(False)
    model.eval()
    return model


class EpisodeEnv:
    """One rank-local MT1 environment with deterministic matched task streams."""

    def __init__(self, *, rank: int, seed: int, task_items: list[tuple[str, str]]) -> None:
        import metaworld
        import numpy as np

        metaworld.register_mw_envs()
        self.rank = rank
        self.rng = np.random.default_rng(seed + PROTOCOL.rank_seed_multiplier * rank)
        self.tasks = task_items
        self.env: Any | None = None
        self.task_index: int | None = None
        self.trial_id: int | None = None
        self.steps = 0
        self.episodes = 0
        self.task_hist = np.zeros(len(self.tasks), dtype=np.int64)

    @staticmethod
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
        model.cam_pos[2] = [0.75, 0.075, 0.7]

    @staticmethod
    def _find_selector(env: Any) -> Any:
        from metaworld.wrappers import RandomTaskSelectWrapper

        current = env
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, RandomTaskSelectWrapper):
                return current
            current = getattr(current, "env", None)
        raise RuntimeError("MetaWorld RandomTaskSelectWrapper is missing")

    def reset(self) -> dict[str, Any]:
        import gymnasium as gym
        import numpy as np

        if self.env is not None:
            self.env.close()
        if self.episodes == 0:
            self.task_index = self.rank % len(self.tasks)
        else:
            self.task_index = int(self.rng.integers(0, len(self.tasks)))
        self.trial_id = int(self.rng.integers(0, PROTOCOL.variants_per_task))
        env_name, _ = self.tasks[self.task_index]
        self.env = gym.make(
            "Meta-World/MT1",
            env_name=env_name,
            seed=int(self.rng.integers(2**31)),
            render_mode="rgb_array",
            camera_id=PROTOCOL.camera_id,
            disable_env_checker=True,
        )
        self._set_camera(self.env)
        selector = self._find_selector(self.env)
        selector.toggle_sample_tasks_on_reset(False)
        if self.trial_id >= len(selector.tasks):
            self.trial_id = 0
        self.env.unwrapped.set_task(selector.tasks[self.trial_id])
        observation, _ = self.env.reset(seed=int(self.rng.integers(2**31)))
        for _ in range(PROTOCOL.reset_settle_steps):
            observation, _, _, _, _ = self.env.step(
                np.zeros(PROTOCOL.physical_action_dims, dtype=np.float32)
            )
        self.steps = 0
        self.episodes += 1
        self.task_hist[self.task_index] += 1
        return self._observation(observation)

    def _observation(self, observation: Any) -> dict[str, Any]:
        import numpy as np
        import torch

        if self.env is None or self.task_index is None:
            raise RuntimeError("environment is not reset")
        image = np.asarray(self.env.render())[::-1, ::-1]
        return {
            "main_images": torch.from_numpy(np.ascontiguousarray(image))[None],
            "states": torch.from_numpy(np.asarray(observation[:4], dtype=np.float32))[None],
            "task_descriptions": [self.tasks[self.task_index][1]],
            "wrist_images": None,
            "extra_view_images": None,
        }

    def step_chunk(self, actions: Any) -> tuple[dict[str, Any], bool, Any]:
        import numpy as np

        if self.env is None:
            raise RuntimeError("environment is not reset")
        action_array = actions.detach().float().cpu().numpy()[0]
        if action_array.shape != (PROTOCOL.execution_prefix, PROTOCOL.physical_action_dims):
            raise ValueError(f"invalid action chunk shape: {action_array.shape}")
        last_observation = None
        executed: list[np.ndarray] = []
        done = False
        for action in action_array:
            observation, _, terminated, truncated, info = self.env.step(
                action.astype(np.float32, copy=True)
            )
            executed.append(action.astype(np.float32, copy=True))
            self.steps += 1
            last_observation = observation
            if terminated or truncated or info.get("success", 0):
                done = True
                break
        if self.steps >= PROTOCOL.episode_limit:
            done = True
        if last_observation is None:
            raise RuntimeError("MetaWorld did not execute an action")
        return self._observation(last_observation), done, np.asarray(executed, dtype=np.float32)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


def _to_device(obs: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in obs.items()
    }


def _rollout(
    model: Any, obs: dict[str, Any], *, seed: int, device: Any
) -> tuple[Any, dict[str, Any]]:
    import torch

    _seed_all(seed)
    with torch.no_grad():
        actions, result = model.predict_action_batch(
            env_obs=_to_device(obs, device), mode="eval", compute_values=False
        )
    expected = (1, PROTOCOL.execution_prefix, PROTOCOL.physical_action_dims)
    if tuple(actions.shape) != expected:
        raise RuntimeError(f"MetaWorld action shape drift: {tuple(actions.shape)} != {expected}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("policy produced non-finite actions")
    chains = result["forward_inputs"].get("chains")
    expected_chains = (
        1,
        PROTOCOL.student_steps + 1,
        PROTOCOL.model_horizon,
        PROTOCOL.model_action_dims,
    )
    if chains is None or tuple(chains.shape) != expected_chains:
        actual_chains = None if chains is None else tuple(chains.shape)
        raise RuntimeError(f"chain shape drift: {actual_chains} != {expected_chains}")
    return actions, result


def _dagger_loss(
    student: Any, student_result: dict[str, Any], teacher_result: dict[str, Any]
) -> Any:
    from path_opd.adapters.openpi import OpenPIAdapter

    adapter = OpenPIAdapter.from_rollout(
        student,
        student_result,
        steps=PROTOCOL.student_steps,
        nft_forward_type=_forward_type().NFT,
        sft_forward_type=_forward_type().SFT,
    )
    return adapter.endpoint_dagger_loss(teacher_result["model_actions"])


class _PathLossRuntime:
    """Own the training-lifetime Path-OPD objective and frozen teacher.

    The rollout context is the only per-update input, so keep the immutable
    teacher service and objective alive for the run while rebinding that
    context before each supervision call.  The large teacher uses the fast
    identity/version guard; byte-level hashing is reserved for offline audits.
    """

    def __init__(self, teacher: Any) -> None:
        from path_opd import ActionContract, FlowSchedule, PathOPD
        from path_opd.adapters.openpi import OpenPIAdapter
        from path_opd.core import FrozenTeacher

        self._forward_inputs: Any | None = None
        self._nft_forward_type = _forward_type().NFT
        self.objective = PathOPD(
            ActionContract(PROTOCOL.execution_prefix, PROTOCOL.physical_action_dims),
            FlowSchedule.uniform(PROTOCOL.student_steps),
        )

        def query(model: Any, states: Any, times: Any) -> Any:
            if self._forward_inputs is None:
                raise RuntimeError("Path-OPD teacher context was not bound")
            return OpenPIAdapter(
                model=model,
                forward_inputs=self._forward_inputs,
                steps=PROTOCOL.teacher_steps,
                nft_forward_type=self._nft_forward_type,
            ).velocity(states, times)

        self.frozen_teacher = FrozenTeacher(
            teacher, query, deep_fingerprint=False
        )

    def bind(self, student_result: dict[str, Any]) -> None:
        """Bind the current rollout context for the next teacher query."""
        forward_inputs = student_result.get("forward_inputs")
        if not isinstance(forward_inputs, Mapping):
            raise ValueError("OpenPI rollout result lacks forward_inputs")
        self._forward_inputs = forward_inputs


def _path_loss(student: Any, student_result: dict[str, Any], runtime: _PathLossRuntime) -> Any:
    from path_opd.adapters.openpi import OpenPIAdapter

    runtime.bind(student_result)
    student_adapter = OpenPIAdapter.from_rollout(
        student,
        student_result,
        steps=PROTOCOL.student_steps,
        nft_forward_type=_forward_type().NFT,
    )
    result = runtime.objective.supervise(
        student_adapter.chains,
        student_adapter.velocity,
        runtime.frozen_teacher,
    )
    return result.loss


def _forward_type() -> Any:
    from rlinf.models.embodiment.base_policy import ForwardType

    return ForwardType


def _allreduce_grads(model: Any, trainable_names: tuple[str, ...], world_size: int) -> None:
    import torch
    import torch.distributed as dist

    parameters = dict(model.named_parameters())
    for name in trainable_names:
        parameter = parameters[name]
        if parameter.grad is None:
            raise RuntimeError(f"missing action-expert gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"non-finite action-expert gradient: {name}")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def _checkpoint(
    *,
    paths: RuntimePaths,
    args: argparse.Namespace,
    model: Any,
    optimizer: Any,
    trainable_names: tuple[str, ...],
    initial_student_sha256: str,
    teacher_sha256: str,
    task_hist: list[int],
    update: int,
    chunks: int,
    release_assets: Mapping[str, str],
    release_code: Mapping[str, str | None],
) -> Path:
    import hashlib

    import safetensors.torch
    import torch

    output = paths.output / "checkpoints" / f"chunks_{chunks:06d}"
    output.mkdir(parents=True, exist_ok=True)
    model_hash = _module_sha256(model)
    safetensors.torch.save_model(model, str(output / "model.safetensors"))
    shutil.copy2(paths.norm_stats, output / "norm_stats.json")
    norm_bundle = output / "lerobot" / "metaworld_mt50"
    norm_bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.norm_stats, norm_bundle / "norm_stats.json")
    release_manifest = _release_checkpoint_identity(
        paths=paths,
        args=args,
        update=update,
        assets=release_assets,
        code=release_code,
    )
    torch.save(
        {
            "model": {
                key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "update": update,
            "chunks": chunks,
            "release_contract": release_manifest,
        },
        output / "trainer_state.pt",
    )
    safetensors.torch.load_model(model, str(output / "model.safetensors"), strict=False)
    reloaded_hash = _module_sha256(model)
    if reloaded_hash != model_hash:
        raise RuntimeError("checkpoint save/reload changed the student model")
    trainable_name_sha256 = hashlib.sha256("\n".join(trainable_names).encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "method": args.method,
        "seed": args.seed,
        "world_size": PROTOCOL.world_size,
        "config_name": PROTOCOL.config_name,
        "model_horizon": PROTOCOL.model_horizon,
        "execution_prefix": PROTOCOL.execution_prefix,
        "physical_action_dims": PROTOCOL.physical_action_dims,
        "model_action_dims": PROTOCOL.model_action_dims,
        "student_steps": PROTOCOL.student_steps,
        "teacher_steps": PROTOCOL.teacher_steps,
        "episode_limit": PROTOCOL.episode_limit,
        "chunks": chunks,
        "optimizer_update": update,
        "normalizer_sha256": file_sha256(paths.norm_stats),
        "initial_student_sha256": initial_student_sha256,
        "teacher_sha256": teacher_sha256,
        "student_sha256": model_hash,
        "reload_student_sha256": reloaded_hash,
        "reload_bitwise_equivalent": True,
        "model_file_sha256": file_sha256(output / "model.safetensors"),
        "trainable_parameter_count": len(trainable_names),
        "trainable_parameter_names_sha256": trainable_name_sha256,
        "task_episode_histogram": task_hist,
        "resume_semantics": "model_optimizer_counters_only_new_environment_stream",
        "private_paths_omitted": True,
        "release_contract": release_manifest,
    }
    atomic_write_json(output / "manifest.json", manifest)
    return output


def _run(args: argparse.Namespace, paths: RuntimePaths) -> int:
    validate_rlinf_host_support(
        paths.rlinf_checkout,
        benchmark="metaworld_mt50",
        patch_path=Path(__file__).resolve().parents[2]
        / "integrations"
        / "rlinf"
        / "path-opd-host-support.patch",
        require_openpi_distribution=True,
    )
    import numpy as np
    import torch
    import torch.distributed as dist

    from path_opd import configure_action_expert, validate_exact_trace

    activate_rlinf(paths.rlinf_checkout)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "MetaWorld training requires CUDA; use --dry-run for CPU-only validation"
        )
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if rank < 0 or local_rank < 0 or world_size != PROTOCOL.world_size:
        raise RuntimeError(
            f"launch with torchrun and exactly {PROTOCOL.world_size} ranks; "
            f"got RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    env: EpisodeEnv | None = None
    try:
        task_items = validate_task_config(paths.task_config)
        release_assets = _release_asset_hashes(paths)
        release_code = _release_code_provenance(paths)
        output_error: list[str | None] = [None]
        if rank == 0:
            try:
                if args.resume is None and paths.output.exists() and any(paths.output.iterdir()):
                    raise ValueError(
                        "training output already contains files; choose a new output or --resume"
                    )
                paths.output.mkdir(parents=True, exist_ok=True)
            except (OSError, ValueError) as error:
                output_error[0] = str(error)
        dist.broadcast_object_list(output_error, src=0)
        if output_error[0] is not None:
            raise RuntimeError(output_error[0])
        _seed_all(args.seed + rank)
        student = _load_model(
            paths.base_model, paths.norm_stats, train_expert_only=True, device=device
        )
        trainable_names = configure_action_expert(student)
        teacher = _load_model(
            paths.teacher_model, paths.norm_stats, train_expert_only=False, device=device
        )
        teacher_hash = _module_sha256(teacher)
        path_runtime = _PathLossRuntime(teacher) if args.method == "path_opd" else None
        if args.resume is not None and args.resume_manifest.get("teacher_sha256") != teacher_hash:
            raise RuntimeError("resume checkpoint was trained with a different teacher")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in student.parameters() if parameter.requires_grad],
            lr=7.91e-6,
            betas=(0.9, 0.95),
            eps=1e-5,
            weight_decay=0.01,
        )
        env = EpisodeEnv(rank=rank, seed=args.seed, task_items=task_items)
        obs = env.reset()
        initial_identity = (rank, int(env.task_index), int(env.trial_id))
        identities: list[Any] = [None] * PROTOCOL.world_size
        dist.all_gather_object(identities, initial_identity)
        if (
            args.max_updates is not None
            and len({item[1] for item in identities}) != PROTOCOL.world_size
        ):
            raise RuntimeError(f"rank task streams are not distinct: {identities}")
        if rank == 0:
            (paths.output / "logs").mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                paths.output / "logs" / "initial_rank_identities.json",
                {"identities": identities, "private_paths_omitted": True},
            )
        dist.barrier()
        update = 0
        chunks = 0
        if args.resume is not None:
            payload = torch.load(
                args.resume / "trainer_state.pt", map_location="cpu", weights_only=False
            )
            payload_release = payload.get("release_contract")
            if payload_release != args.resume_manifest.get("release_contract"):
                raise RuntimeError("trainer state and sidecar release manifests differ")
            student.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            update = int(payload["update"])
            chunks = int(payload["chunks"])
            if chunks != update * PROTOCOL.world_size:
                raise RuntimeError("resume checkpoint has inconsistent chunk/update counters")
            if update >= args.target_chunks // PROTOCOL.world_size:
                raise RuntimeError("resume checkpoint is already at or beyond target-chunks")
        initial_student_hash = _module_sha256(student)
        planned_updates = args.target_chunks // PROTOCOL.world_size
        target_updates = planned_updates if args.max_updates is None else args.max_updates
        while update < target_updates:
            update += 1
            student.eval()
            rollout_seed = args.seed + rank * PROTOCOL.rank_seed_multiplier + update
            student_actions, student_result = _rollout(
                student, obs, seed=rollout_seed, device=device
            )
            teacher_result = None
            if args.method == "endpoint_dagger":
                _, teacher_result = _rollout(teacher, obs, seed=rollout_seed, device=device)
            next_obs, done, executed = env.step_chunk(student_actions)
            recorded = torch.as_tensor(executed)
            submitted = student_actions.detach().float().cpu()[:, : executed.shape[0]]
            trace_audit = validate_exact_trace(
                student_result["forward_inputs"]["chains"],
                student_result["model_actions"],
                submitted,
                recorded[None],
                expected_steps=PROTOCOL.student_steps,
            )
            optimizer.zero_grad(set_to_none=True)
            student.train()
            if args.method == "endpoint_dagger":
                if teacher_result is None:
                    raise AssertionError("Endpoint-DAgger teacher rollout is missing")
                loss = _dagger_loss(student, student_result, teacher_result)
            else:
                if path_runtime is None:
                    raise AssertionError("Path-OPD runtime is missing")
                loss = _path_loss(student, student_result, path_runtime)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            _allreduce_grads(student, trainable_names, PROTOCOL.world_size)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in student.parameters() if parameter.requires_grad],
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            torch.cuda.synchronize()
            chunks = update * PROTOCOL.world_size
            record = {
                "rank": rank,
                "update": update,
                "chunks": chunks,
                "loss": float(loss.detach().cpu()),
                "task_index": int(env.task_index),
                "trial_id": int(env.trial_id),
                "episodes": env.episodes,
                "task_hist": env.task_hist.tolist(),
                "trace": trace_audit,
                "k": PROTOCOL.student_steps,
                "horizon": PROTOCOL.model_horizon,
                "execution_prefix": PROTOCOL.execution_prefix,
                "physical_action_dims": PROTOCOL.physical_action_dims,
                "episode_limit": PROTOCOL.episode_limit,
            }
            with (paths.output / "logs" / f"rank_{rank:02d}.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            if done:
                next_obs = env.reset()
            obs = next_obs
            checkpoint_due = chunks in {10_000, 20_000, 40_000, 60_000, 80_000} or (
                update == target_updates
            )
            if checkpoint_due:
                gathered: list[Any] = [None] * PROTOCOL.world_size
                dist.all_gather_object(gathered, env.task_hist.tolist())
                total_hist = np.sum(np.asarray(gathered), axis=0).astype(int).tolist()
                if rank == 0:
                    _checkpoint(
                        paths=paths,
                        args=args,
                        model=student,
                        optimizer=optimizer,
                        trainable_names=trainable_names,
                        initial_student_sha256=initial_student_hash,
                        teacher_sha256=teacher_hash,
                        task_hist=total_hist,
                        update=update,
                        chunks=chunks,
                        release_assets=release_assets,
                        release_code=release_code,
                    )
                dist.barrier()
        if _module_sha256(teacher) != teacher_hash:
            raise RuntimeError("frozen teacher changed during training")
        if rank == 0:
            if args.max_updates is not None:
                status = "QUALIFICATION_PASS"
            elif args.target_chunks == PROTOCOL.target_chunks:
                status = "PAPER_RUN_COMPLETE"
            else:
                status = "CUSTOM_RUN_COMPLETE"
            atomic_write_json(
                paths.output / "progress.json",
                {
                    "status": status,
                    "method": args.method,
                    "chunks": chunks,
                    "update": update,
                    "teacher_sha256": teacher_hash,
                    "private_paths_omitted": True,
                },
            )
        dist.barrier()
    finally:
        if env is not None:
            env.close()
        dist.destroy_process_group()
    return 0


def _synthetic_smoke(argv: list[str]) -> int:
    """Run the synthetic MetaWorld contract through the public worker entrypoint."""

    parser = argparse.ArgumentParser(
        prog="metaworld-train --synthetic-smoke",
        description=(
            "Run the synthetic MetaWorld runner contract; no RLinf checkout, "
            "model, simulator, CUDA device, or benchmark asset is used."
        ),
    )
    parser.add_argument("--synthetic-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("artifacts/runner-smoke/metaworld_mt50")
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
        "metaworld_mt50",
        args.work_dir,
        entrypoint="benchmarks/metaworld_mt50/train.py",
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
            print(f"metaworld-train synthetic smoke: {error}", file=sys.stderr)
            return 2
    args = build_parser().parse_args(raw_argv)
    try:
        paths = _validate_args(args)
        if args.dry_run:
            print_json(
                dry_run_plan(
                    "train",
                    paths,
                    method=args.method,
                    seed=args.seed,
                    target_chunks=args.target_chunks,
                    max_updates=args.max_updates,
                )
            )
            return 0
        return _run(args, paths)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"metaworld-train: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

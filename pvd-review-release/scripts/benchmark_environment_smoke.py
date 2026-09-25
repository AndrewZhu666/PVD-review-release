#!/usr/bin/env python3
"""Run one-step environment checks for the released benchmark runners.

This script deliberately sits below model training and above dependency-free
``--dry-run`` checks.  It accepts every external location explicitly, imports
the selected RLinf checkout only after path validation, resets one environment,
and executes a small zero-action rollout.  A missing dependency, simulator asset,
or CUDA runtime is reported as ``BLOCKED``; an exception after all prerequisites
are present is reported as ``FAIL``.  The default command is therefore useful
on a partially provisioned review machine and does not pretend that a blocked
check passed.

The smoke does not download, copy, hash recursively, or modify caller assets.
It is an environment qualification check, not a model-training or paper-result
reproduction command.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Running this file directly from a source checkout should still find the
# release package.  The path is derived from the script location; no host
# specific fallback or environment variable is consulted.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
for _path in (_REPOSITORY_ROOT, _SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SCHEMA = "path-opd-benchmark-environment-smoke-v1"
EXPECTED_RLINF_REVISION = "fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c"
BENCHMARKS = ("maniskill", "calvin_abc_d", "metaworld_mt50")
STATUSES = ("PASS", "BLOCKED", "FAIL")


class EnvironmentBlocked(RuntimeError):
    """A prerequisite is unavailable on the machine running the smoke."""


@dataclass(frozen=True)
class SmokeArgs:
    benchmarks: tuple[str, ...]
    rlinf_checkout: Path | None
    maniskill_simulator_assets: Path | None
    maniskill_package_assets: Path | None
    calvin_environment_assets: Path | None
    calvin_task_oracle_annotations: Path | None
    metaworld_task_config: Path | None
    metaworld_assets: Path | None
    steps: int
    seed: int
    require_cuda: bool
    require_pinned_rlinf: bool
    cpu_simulator_contract: bool
    skip_algorithm_smoke: bool = False
    verbose: bool = False


def _path_text(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser().resolve(strict=False)


def _path_record(path: Path | None, *, kind: str, required: bool) -> dict[str, Any]:
    """Describe one caller-supplied path without recursively inspecting it."""

    resolved = _resolve(path)
    if resolved is None:
        return {
            "supplied": False,
            "required": required,
            "kind": kind,
            "exists": False,
            "path": None,
        }
    exists = resolved.is_file() if kind == "file" else resolved.is_dir()
    return {
        "supplied": True,
        "required": required,
        "kind": kind,
        "exists": exists,
        "path": str(resolved),
    }


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _required_files(checkout: Path, benchmark: str) -> tuple[str, ...]:
    common = ("rlinf/models/embodiment/base_policy.py",)
    if benchmark == "maniskill":
        return (
            *common,
            "rlinf/envs/maniskill/maniskill_env.py",
            "rlinf/envs/maniskill/maniskill_offload_env.py",
        )
    if benchmark == "calvin_abc_d":
        return (
            *common,
            "rlinf/envs/calvin/__init__.py",
            "rlinf/envs/calvin/calvin_gym_env.py",
            "rlinf/envs/calvin/calvin_cfg/.hydra/merged_config.yaml",
            "rlinf/envs/calvin/calvin_cfg/.hydra/calvin_scene_D.yaml",
        )
    if benchmark == "metaworld_mt50":
        # RLinf has shipped both layouts.  Prefer the one actually present in
        # the selected checkout so a valid sim/ checkout is not rejected by a
        # stale legacy path.
        task_config_candidates = (
            "rlinf/envs/metaworld/metaworld_config.json",
            "rlinf/envs/sim/metaworld/metaworld_config.json",
        )
        task_config = next(
            (relative for relative in task_config_candidates if (checkout / relative).is_file()),
            task_config_candidates[0],
        )
        return (
            *common,
            task_config,
        )
    raise ValueError(f"unknown benchmark: {benchmark}")


def _git_revision(checkout: Path) -> str | None:
    """Read a selected checkout's revision without inspecting its file tree."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    revision = completed.stdout.strip()
    valid = len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
    return revision if valid else None


def _locate_metaworld_config(checkout: Path | None) -> Path | None:
    if checkout is None:
        return None
    for relative in (
        "rlinf/envs/metaworld/metaworld_config.json",
        "rlinf/envs/sim/metaworld/metaworld_config.json",
    ):
        candidate = checkout / relative
        if candidate.is_file():
            return candidate.resolve()
    return None


def _preflight(args: SmokeArgs, benchmark: str) -> tuple[dict[str, Any], list[str]]:
    """Validate explicit paths and return a report plus blocking reasons."""

    checks: dict[str, Any] = {}
    blockers: list[str] = []

    checkout = _resolve(args.rlinf_checkout)
    checkout_record = _path_record(checkout, kind="directory", required=True)
    checks["rlinf_checkout"] = checkout_record
    if checkout is None:
        blockers.append("missing explicit --rlinf-checkout/--rlinf-root")
    elif not checkout.is_dir():
        blockers.append(f"RLinf checkout is not a directory: {checkout}")
    else:
        observed_revision = _git_revision(checkout)
        checks["rlinf_revision"] = {
            "expected": EXPECTED_RLINF_REVISION,
            "observed": observed_revision,
            "verified": observed_revision == EXPECTED_RLINF_REVISION,
            "verification_scope": "metadata_only",
        }
        if args.require_pinned_rlinf:
            if observed_revision is None:
                blockers.append(
                    "RLinf revision could not be verified; use a Git checkout or omit "
                    "--require-pinned-rlinf for a presence-only probe"
                )
            elif observed_revision != EXPECTED_RLINF_REVISION:
                blockers.append(
                    "RLinf revision mismatch: expected "
                    f"{EXPECTED_RLINF_REVISION}, observed {observed_revision}"
                )
        missing = [
            relative
            for relative in _required_files(checkout, benchmark)
            if not (checkout / relative).is_file()
        ]
        checks["rlinf_required_files"] = {"missing": missing, "passed": not missing}
        if missing:
            blockers.append("RLinf checkout is missing: " + ", ".join(missing))
        if args.require_pinned_rlinf:
            try:
                from path_opd.release_contract import validate_rlinf_host_support

                checks["rlinf_host_support"] = validate_rlinf_host_support(
                    checkout,
                    benchmark=benchmark,
                    patch_path=(
                        _REPOSITORY_ROOT
                        / "integrations"
                        / "rlinf"
                        / "path-opd-host-support.patch"
                    ),
                    require_openpi_distribution=True,
                )
            except (ImportError, OSError, ValueError) as error:
                checks["rlinf_host_support"] = {"verified": False}
                blockers.append(f"RLinf host integration is not qualified: {error}")

    if benchmark == "maniskill":
        for label, path in (
            ("maniskill_simulator_assets", args.maniskill_simulator_assets),
            ("maniskill_package_assets", args.maniskill_package_assets),
        ):
            checks[label] = _path_record(path, kind="directory", required=True)
            if path is None:
                blockers.append(f"missing explicit --{label.replace('_', '-')}")
            elif not (_resolve(path) or Path()).is_dir():
                blockers.append(f"{label} is not a directory: {_resolve(path)}")

    elif benchmark == "calvin_abc_d":
        checks["calvin_environment_assets"] = _path_record(
            args.calvin_environment_assets, kind="directory", required=True
        )
        if args.calvin_environment_assets is None:
            blockers.append("missing explicit --calvin-environment-assets/--calvin-assets")
        elif not (_resolve(args.calvin_environment_assets) or Path()).is_dir():
            blockers.append(
                "calvin_environment_assets is not a directory: "
                f"{_resolve(args.calvin_environment_assets)}"
            )
        annotation_record = _path_record(
            args.calvin_task_oracle_annotations, kind="file", required=False
        )
        checks["calvin_task_oracle_annotations"] = annotation_record
        if args.calvin_task_oracle_annotations is not None and not (
            _resolve(args.calvin_task_oracle_annotations) or Path()
        ).is_file():
            blockers.append(
                "calvin_task_oracle_annotations is not a file: "
                f"{_resolve(args.calvin_task_oracle_annotations)}"
            )

    elif benchmark == "metaworld_mt50":
        task_config = _resolve(args.metaworld_task_config) or _locate_metaworld_config(checkout)
        checks["metaworld_task_config"] = _path_record(task_config, kind="file", required=True)
        if task_config is None:
            blockers.append(
                "missing explicit --metaworld-task-config and no supported config "
                "under RLinf checkout"
            )
        elif not task_config.is_file():
            blockers.append(f"metaworld task config is not a file: {task_config}")
        # MetaWorld normally obtains MuJoCo assets through the installed package.
        # An optional caller path is accepted for sites that keep an explicit cache.
        if args.metaworld_assets is not None:
            checks["metaworld_assets"] = _path_record(
                args.metaworld_assets, kind="directory", required=False
            )
            if not (_resolve(args.metaworld_assets) or Path()).is_dir():
                blockers.append(
                    "metaworld_assets is not a directory: "
                    f"{_resolve(args.metaworld_assets)}"
                )

    return checks, blockers


def _module_available(name: str, checkout: Path | None = None) -> bool:
    if checkout is not None and str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
        importlib.invalidate_caches()
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _dependency_check(benchmark: str, checkout: Path) -> list[str]:
    dependencies = {
        "maniskill": ("torch", "numpy", "omegaconf", "gymnasium", "mani_skill", "rlinf"),
        "calvin_abc_d": (
            "torch",
            "numpy",
            "hydra",
            "omegaconf",
            "gymnasium",
            "calvin_env",
            "calvin_agent",
            "pytorch_lightning",
            "pybullet",
            "rlinf",
        ),
        "metaworld_mt50": ("torch", "numpy", "gymnasium", "mujoco", "metaworld", "rlinf"),
    }[benchmark]
    return [name for name in dependencies if not _module_available(name, checkout)]


def _runtime_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        snapshot.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
            }
        )
    except Exception as error:  # pragma: no cover - dependency-only path
        snapshot["torch_error"] = f"{type(error).__name__}: {error}"
    return snapshot


def _run_algorithm_contract(args: SmokeArgs, benchmark: str) -> dict[str, Any]:
    """Run the released algorithm contract before touching a real simulator.

    The policy is deliberately synthetic and the temporary checkpoint is
    deleted when this function returns. This proves that the benchmark's
    declared flow/action contract, Path-OPD objective, adapter, baseline,
    optimizer, and exact resume are reachable from the same command that will
    probe the external simulator. It is never a benchmark score.
    """

    from path_opd.adapters.benchmark_smoke import (
        SyntheticSmokeConfig,
        run_benchmark_smoke,
    )

    with tempfile.TemporaryDirectory(prefix="path-opd-algorithm-smoke-") as directory:
        report = run_benchmark_smoke(
            benchmark,
            Path(directory),
            SyntheticSmokeConfig(
                updates=max(2, min(args.steps + 1, 4)),
                batch_size=1,
                seed=args.seed,
            ),
        )
    uninterrupted = report.get("uninterrupted", {})
    return {
        "schema": report["schema"],
        "status": report["status"],
        "claim": report["claim"],
        "synthetic_only": report["synthetic_only"],
        "attempted": True,
        "completed": True,
        "executed": True,
        "external_runtime_executed": False,
        "external_assets_used": report["external_assets_used"],
        "external_openpi_executed": report["external_openpi_executed"],
        "real_simulator": report["real_simulator"],
        "gpu_executed": report["gpu_executed"],
        "paper_scale": report["paper_scale"],
        "config": report["config"],
        "contract": report["contract"],
        "checks": report["checks"],
        "exact_resume": report["interrupted"]["exact_resume"],
        "deterministic_digests": {
            key: uninterrupted[key]
            for key in (
                "student_sha256",
                "optimizer_sha256",
                "generator_sha256",
                "teacher_sha256",
            )
            if key in uninterrupted
        },
    }


def _shape(value: Any) -> Any:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return [int(item) for item in shape]
        except (TypeError, ValueError):
            return str(shape)
    if isinstance(value, Mapping):
        return {str(key): _shape(item) for key, item in list(value.items())[:24]}
    if isinstance(value, (list, tuple)):
        return {"length": len(value), "first": _shape(value[0]) if value else None}
    return type(value).__name__


def _activate_rlinf(checkout: Path) -> None:
    checkout = checkout.resolve()
    if not (checkout / "rlinf").is_dir():
        raise EnvironmentBlocked(f"selected checkout has no rlinf/ package: {checkout}")
    existing = sys.modules.get("rlinf")
    if existing is not None:
        module_file = getattr(existing, "__file__", None)
        if module_file is None or not _within(Path(module_file).resolve(), checkout):
            raise RuntimeError("rlinf was already imported from a different checkout")
    sys.path.insert(0, str(checkout))
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("rlinf")
    except (ImportError, ModuleNotFoundError) as error:
        raise EnvironmentBlocked(f"selected RLinf checkout cannot be imported: {error}") from error
    module_file = getattr(module, "__file__", None)
    if module_file is None or not _within(Path(module_file).resolve(), checkout):
        raise RuntimeError("selected RLinf checkout did not win Python import resolution")


def _run_maniskill(args: SmokeArgs) -> dict[str, Any]:
    checkout = _resolve(args.rlinf_checkout)
    if checkout is None:  # guarded by preflight
        raise EnvironmentBlocked("RLinf checkout was not supplied")
    _activate_rlinf(checkout)
    simulator_assets = _resolve(args.maniskill_simulator_assets)
    package_assets = _resolve(args.maniskill_package_assets)
    if simulator_assets is None or package_assets is None:
        raise EnvironmentBlocked("ManiSkill asset paths were not supplied")
    bridge_candidates = (
        simulator_assets / "data/tasks/bridge_v2_real2sim_dataset",
        package_assets / "data/tasks/bridge_v2_real2sim_dataset",
    )
    if not any(candidate.exists() for candidate in bridge_candidates):
        raise EnvironmentBlocked(
            "ManiSkill formal task asset bridge_v2_real2sim is missing; "
            "refusing an interactive asset download"
        )
    os.environ["MANISKILL_ASSET_DIR"] = str(simulator_assets)
    os.environ["MS_ASSET_DIR"] = str(package_assets)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.maniskill.maniskill_offload_env import _ManiskillEnvCore

    from benchmarks.maniskill.train import _environment_config

    config = _environment_config(OmegaConf)
    config.seed = args.seed
    mode = "formal_gpu_config"
    if args.cpu_simulator_contract:
        # This is intentionally a separate contract.  It checks that the
        # caller's ManiSkill install can construct the task on a CPU backend;
        # it is not the paper runner, whose released config uses GPU physics.
        config.init_params.sim_backend = "cpu"
        config.init_params.render_mode = "rgb_array"
        mode = "cpu_simulator_contract"
    elif not torch.cuda.is_available():
        raise EnvironmentBlocked(
            "ManiSkill release config requests sim_backend=gpu, but "
            "torch.cuda.is_available() is false"
        )
    env = _ManiskillEnvCore(
        cfg=config,
        num_envs=1,
        seed_offset=args.seed,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )
    try:
        # Training deliberately disables metrics; preserve the same compatibility shim.
        env._record_metrics = lambda _reward, infos: infos
        observation, _ = env.reset()
        action = torch.zeros(
            (1, 7), dtype=torch.float32, device=env.device
        )
        step_records: list[dict[str, Any]] = []
        for index in range(args.steps):
            next_observation, reward, terminated, truncated, _info = env.step(
                action, auto_reset=False
            )
            step_records.append(
                {
                    "step": index + 1,
                    "observation": _shape(next_observation),
                    "reward": _shape(reward),
                    "terminated": _shape(terminated),
                    "truncated": _shape(truncated),
                }
            )
        return {
            "reset_observation": _shape(observation),
            "action_shape": [1, 7],
            "device": str(env.device),
            "steps_completed": len(step_records),
            "steps": step_records,
            "mode": mode,
            "protocol_config": mode,
            "config_matches_formal_protocol": mode == "formal_gpu_config",
            "formal_benchmark": False,
            "requested_steps": args.steps,
            "one_step_smoke_only": args.steps == 1,
            "short_rollout_smoke_only": args.steps > 1,
        }
    finally:
        env.close()


def _run_calvin(args: SmokeArgs) -> dict[str, Any]:
    checkout = _resolve(args.rlinf_checkout)
    assets = _resolve(args.calvin_environment_assets)
    if checkout is None or assets is None:
        raise EnvironmentBlocked("CALVIN checkout and environment-assets paths are required")
    _activate_rlinf(checkout)
    os.environ.setdefault("EGL_VISIBLE_DEVICES", "0")

    import numpy as np

    from benchmarks.calvin_abc_d.common import PROTOCOL
    from benchmarks.calvin_abc_d.environment import CalvinEnvironmentAdapter

    # Evaluation mode uses one raw scene so this check does not silently invent
    # a task-oracle/panel.  The full runner separately validates those inputs.
    adapter = CalvinEnvironmentAdapter(
        rlinf_checkout=checkout,
        environment_assets=assets,
        rank=0,
        world_size=1,
        seed=args.seed,
        evaluation=True,
    )
    try:
        observation = adapter.reset()
        action = np.zeros(PROTOCOL.physical_action_dims, dtype=np.float32)
        # CALVIN's controller treats the gripper command as a binary value;
        # zero is rejected by the official environment assertion.
        action[-1] = 1.0
        records: list[dict[str, Any]] = []
        for index in range(args.steps):
            next_observation, reward, done, info = adapter.step(action)
            records.append(
                {
                    "step": index + 1,
                    "observation": _shape(next_observation),
                    "reward": _shape(reward),
                    "done": bool(done),
                    "info": _shape(info),
                }
            )
        return {
            "reset_observation": _shape(observation),
            "action_shape": [PROTOCOL.physical_action_dims],
            "steps_completed": len(records),
            "steps": records,
            "scene": PROTOCOL.evaluation_scene,
            "task_oracle_annotations_supplied": args.calvin_task_oracle_annotations is not None,
            "requested_steps": args.steps,
            "one_step_smoke_only": args.steps == 1,
            "short_rollout_smoke_only": args.steps > 1,
            "formal_benchmark": False,
        }
    finally:
        adapter.close()


def _run_metaworld(args: SmokeArgs) -> dict[str, Any]:
    checkout = _resolve(args.rlinf_checkout)
    task_config = _resolve(args.metaworld_task_config) or _locate_metaworld_config(checkout)
    if checkout is None or task_config is None:
        raise EnvironmentBlocked("MetaWorld checkout and task-config paths are required")

    # MetaWorld renders an RGB observation during reset/action.  On a review
    # machine there is often no X11 display, and MuJoCo otherwise defaults to
    # GLX before the environment has a chance to report a useful error.  Bind
    # the standard headless backend before importing MetaWorld/MuJoCo, while
    # preserving an explicitly selected backend from the caller.
    render_backend = _prepare_metaworld_headless_rendering()
    _activate_rlinf(checkout)

    import torch

    from benchmarks.metaworld_mt50 import common
    from benchmarks.metaworld_mt50.train import EpisodeEnv

    tasks = common.validate_task_config(task_config)
    env = EpisodeEnv(rank=0, seed=args.seed, task_items=tasks)
    try:
        observation = env.reset()
        action = torch.zeros(
            (1, common.PROTOCOL.execution_prefix, common.PROTOCOL.physical_action_dims)
        )
        records: list[dict[str, Any]] = []
        for index in range(args.steps):
            next_observation, done, executed = env.step_chunk(action)
            records.append(
                {
                    "step": index + 1,
                    "observation": _shape(next_observation),
                    "done": bool(done),
                    "executed_action": _shape(executed),
                }
            )
            if done and index + 1 < args.steps:
                env.reset()
        return {
            "reset_observation": _shape(observation),
            "action_shape": [
                1,
                common.PROTOCOL.execution_prefix,
                common.PROTOCOL.physical_action_dims,
            ],
            "steps_completed": len(records),
            "steps": records,
            "task_count": len(tasks),
            "task_index": env.task_index,
            "trial_id": env.trial_id,
            "render_backend": render_backend,
            "requested_steps": args.steps,
            "one_step_smoke_only": args.steps == 1,
            "short_rollout_smoke_only": args.steps > 1,
            "formal_benchmark": False,
        }
    finally:
        env.close()


def _prepare_metaworld_headless_rendering() -> str:
    """Select EGL for headless MetaWorld smoke unless the caller opted in.

    MuJoCo reads ``MUJOCO_GL`` while importing/creating its first context, so
    this must run before importing ``metaworld``.  A return value is included
    in the smoke report to make the selected rendering contract auditable.
    """

    backend = os.environ.get("MUJOCO_GL")
    if not backend and not os.environ.get("DISPLAY"):
        backend = "egl"
        os.environ["MUJOCO_GL"] = backend
        source = "auto_egl_no_display"
    else:
        source = "caller"
    if backend and backend.lower() == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return f"{backend or 'mujoco_default'}:{source}"


def _is_graphics_context_error(error: BaseException) -> bool:
    """Recognize missing headless graphics as an environment block.

    Import/dependency failures are handled separately above.  These markers
    are deliberately narrow so arbitrary simulator or runner exceptions still
    remain actionable ``FAIL`` results.
    """

    current: BaseException | None = error
    messages: list[str] = []
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__
    text = " ".join(messages)
    return any(
        marker in text
        for marker in (
            "opengl platform library has not been loaded",
            "mjr_makecontext",
            "x11: the display environment variable is missing",
            "the glfw library is not initialized",
        )
    )


def _is_missing_simulator_asset_error(error: BaseException) -> bool:
    """Recognize a simulator that started but lacks its external asset bundle.

    A caller can supply an existing source checkout as the environment path
    while still omitting the large robot/scene files.  PyBullet reports that
    condition as a generic ``Cannot load URDF file`` error, which is a
    prerequisite block rather than a defect in the released runner.  Keep the
    markers intentionally narrow so arbitrary simulator exceptions remain
    actionable ``FAIL`` results.
    """

    current: BaseException | None = error
    messages: list[str] = []
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__
    text = " ".join(messages)
    return "cannot load urdf file" in text or (
        "urdf file '" in text and "not found" in text
    )


def _is_dependency_error(error: BaseException) -> bool:
    """Recognize imports wrapped by an external runner exception."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ImportError, ModuleNotFoundError)):
            return True
        current = current.__cause__ or current.__context__
    return False


_RUNNERS: dict[str, Callable[[SmokeArgs], dict[str, Any]]] = {
    "maniskill": _run_maniskill,
    "calvin_abc_d": _run_calvin,
    "metaworld_mt50": _run_metaworld,
}


def _result(
    *,
    args: SmokeArgs,
    benchmark: str,
    status: str,
    checks: dict[str, Any],
    blockers: list[str],
    evidence: dict[str, Any] | None = None,
    error: BaseException | None = None,
    started: float,
) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(status)
    payload: dict[str, Any] = {
        "benchmark": benchmark,
        "status": status,
        "checks": checks,
        "blockers": blockers,
        "evidence": evidence or {},
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "runtime": _runtime_snapshot(),
    }
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        if args.verbose:
            payload["error"]["traceback"] = traceback.format_exc()
    return payload


def _skipped_algorithm_contract() -> dict[str, Any]:
    """Describe an intentionally skipped synthetic algorithm check."""

    return {
        "status": "SKIPPED",
        "claim": "synthetic_algorithm_contract_skipped",
        "synthetic_only": True,
        "attempted": False,
        "completed": False,
        "executed": False,
        "paper_scale": False,
    }


def _environment_evidence(
    algorithm_contract: dict[str, Any],
    *,
    steps: int,
    attempted: bool,
    executed: bool,
    external_runtime: bool,
    gpu_executed: bool = False,
) -> dict[str, Any]:
    """Attach an explicit boundary record to every environment result."""

    return {
        "algorithm_synthetic_contract": algorithm_contract,
        "environment_probe_attempted": attempted,
        "environment_probe_executed": executed,
        "external_runtime_executed": external_runtime,
        "external_assets_used": external_runtime,
        "real_simulator": external_runtime,
        "gpu_executed": gpu_executed,
        "external_openpi_executed": False,
        "external_training_or_evaluation_executed": False,
        "requested_steps": steps,
        "one_step_smoke_only": steps == 1,
        "short_rollout_smoke_only": steps > 1,
        "formal_benchmark": False,
        "paper_scale": False,
    }


def run(args: SmokeArgs) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for benchmark in args.benchmarks:
        started = time.monotonic()
        if args.skip_algorithm_smoke:
            algorithm_contract = _skipped_algorithm_contract()
        else:
            try:
                algorithm_contract = _run_algorithm_contract(args, benchmark)
            except Exception as error:
                # A failure in the dependency-light released algorithm is a
                # real release failure, even when the external simulator is
                # unavailable.  Keep this distinct from an environment block.
                algorithm_contract = {
                    "status": "FAIL",
                    "claim": "synthetic_contract_smoke_only",
                    "synthetic_only": True,
                    "attempted": True,
                    "completed": False,
                    "executed": False,
                    "paper_scale": False,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
                results.append(
                    _result(
                        args=args,
                        benchmark=benchmark,
                        status="FAIL",
                        checks={"algorithm_synthetic_contract": algorithm_contract},
                        blockers=[
                            "synthetic algorithm contract failed: "
                            f"{type(error).__name__}: {error}"
                        ],
                        evidence=_environment_evidence(
                            algorithm_contract,
                            steps=args.steps,
                            attempted=False,
                            executed=False,
                            external_runtime=False,
                        ),
                        error=error,
                        started=started,
                    )
                )
                continue

        if algorithm_contract.get("status") != "PASS" and not args.skip_algorithm_smoke:
            results.append(
                _result(
                    args=args,
                    benchmark=benchmark,
                    status="FAIL",
                    checks={"algorithm_synthetic_contract": algorithm_contract},
                    blockers=["synthetic algorithm contract did not pass"],
                    evidence=_environment_evidence(
                        algorithm_contract,
                        steps=args.steps,
                        attempted=False,
                        executed=False,
                        external_runtime=False,
                    ),
                    started=started,
                )
            )
            continue

        environment_checks, blockers = _preflight(args, benchmark)
        checks = {
            "algorithm_synthetic_contract": algorithm_contract,
            **environment_checks,
        }
        algorithm_evidence = _environment_evidence(
            algorithm_contract,
            steps=args.steps,
            attempted=False,
            executed=False,
            external_runtime=False,
        )
        if blockers:
            results.append(
                _result(
                    args=args,
                    benchmark=benchmark,
                    status="BLOCKED",
                    checks=checks,
                    blockers=blockers,
                    evidence=algorithm_evidence,
                    started=started,
                )
            )
            continue

        checkout = _resolve(args.rlinf_checkout)
        assert checkout is not None  # established by preflight
        missing_dependencies = _dependency_check(benchmark, checkout)
        checks["dependencies"] = {
            "missing": missing_dependencies,
            "passed": not missing_dependencies,
        }
        if missing_dependencies:
            results.append(
                _result(
                    args=args,
                    benchmark=benchmark,
                    status="BLOCKED",
                    checks=checks,
                    blockers=["missing import(s): " + ", ".join(missing_dependencies)],
                    evidence=algorithm_evidence,
                    started=started,
                )
            )
            continue

        if args.require_cuda:
            runtime = _runtime_snapshot()
            checks["cuda"] = {
                "required": True,
                "available": runtime.get("cuda_available", False),
                "device_count": runtime.get("cuda_device_count", 0),
            }
            if not runtime.get("cuda_available", False):
                results.append(
                    _result(
                        args=args,
                        benchmark=benchmark,
                        status="BLOCKED",
                        checks=checks,
                        blockers=["CUDA was explicitly required but is unavailable"],
                        evidence=algorithm_evidence,
                        started=started,
                    )
                )
                continue
        else:
            checks["cuda"] = {
                "required": False,
                "available": _runtime_snapshot().get("cuda_available"),
            }

        try:
            evidence = _RUNNERS[benchmark](args)
        except EnvironmentBlocked as error:
            blocked_evidence = _environment_evidence(
                algorithm_contract,
                steps=args.steps,
                attempted=True,
                executed=False,
                external_runtime=False,
            )
            results.append(
                _result(
                    args=args,
                    benchmark=benchmark,
                    status="BLOCKED",
                    checks=checks,
                    blockers=[str(error)],
                    evidence=blocked_evidence,
                    error=error,
                    started=started,
                )
            )
        except (ImportError, ModuleNotFoundError) as error:
            blocked_evidence = _environment_evidence(
                algorithm_contract,
                steps=args.steps,
                attempted=True,
                executed=False,
                external_runtime=False,
            )
            results.append(
                _result(
                    args=args,
                    benchmark=benchmark,
                    status="BLOCKED",
                    checks=checks,
                    blockers=[f"runtime import unavailable: {error}"],
                    evidence=blocked_evidence,
                    error=error,
                    started=started,
                )
            )
        except EOFError as error:
            blocked_evidence = _environment_evidence(
                algorithm_contract,
                steps=args.steps,
                attempted=True,
                executed=False,
                external_runtime=False,
            )
            results.append(
                _result(
                    args=args,
                    benchmark=benchmark,
                    status="BLOCKED",
                    checks=checks,
                    blockers=[
                        "environment attempted an interactive asset/download prompt; "
                        "supply the missing asset explicitly"
                    ],
                    evidence=blocked_evidence,
                    error=error,
                    started=started,
                )
            )
        except Exception as error:  # pragma: no cover - exercised by real runtimes
            failed_evidence = _environment_evidence(
                algorithm_contract,
                steps=args.steps,
                attempted=True,
                executed=False,
                external_runtime=False,
            )
            if _is_dependency_error(error):
                results.append(
                    _result(
                        args=args,
                        benchmark=benchmark,
                        status="BLOCKED",
                        checks=checks,
                        blockers=[f"runtime dependency unavailable: {error}"],
                        evidence=failed_evidence,
                        error=error,
                        started=started,
                    )
                )
                continue
            if _is_graphics_context_error(error):
                results.append(
                    _result(
                        args=args,
                        benchmark=benchmark,
                        status="BLOCKED",
                        checks=checks,
                        blockers=[
                            "headless graphics context is unavailable; configure a usable "
                            f"MuJoCo/EGL backend: {error}"
                        ],
                        evidence=failed_evidence,
                        error=error,
                        started=started,
                    )
                )
                continue
            if _is_missing_simulator_asset_error(error):
                results.append(
                    _result(
                        args=args,
                        benchmark=benchmark,
                        status="BLOCKED",
                        checks=checks,
                        blockers=[
                            "simulator asset bundle is incomplete; supply the required "
                            f"external scene/robot files: {error}"
                        ],
                        evidence=failed_evidence,
                        error=error,
                        started=started,
                    )
                )
                continue
            results.append(
                _result(
                    args=args,
                    benchmark=benchmark,
                    status="FAIL",
                    checks=checks,
                    blockers=[],
                    evidence=failed_evidence,
                    error=error,
                    started=started,
                )
            )
        else:
            combined_evidence = dict(evidence)
            combined_evidence.update(
                _environment_evidence(
                    algorithm_contract,
                    steps=args.steps,
                    attempted=True,
                    executed=True,
                    external_runtime=True,
                    gpu_executed=str(evidence.get("device", "")).startswith("cuda"),
                )
            )
            results.append(
                _result(
                    args=args,
                    benchmark=benchmark,
                    status="PASS",
                    checks=checks,
                    blockers=[],
                    evidence=combined_evidence,
                    started=started,
                )
            )

    statuses = [item["status"] for item in results]
    overall = "FAIL" if "FAIL" in statuses else ("BLOCKED" if "BLOCKED" in statuses else "PASS")
    algorithm_enabled = not args.skip_algorithm_smoke
    algorithm_contracts = [
        item["checks"].get("algorithm_synthetic_contract", {}) for item in results
    ]
    algorithm_executed = any(
        bool(contract.get("executed")) for contract in algorithm_contracts
    )
    algorithm_completed = bool(algorithm_contracts) and all(
        contract.get("status") == "PASS" for contract in algorithm_contracts
    )
    environment_evidence = [item.get("evidence", {}) for item in results]
    environment_attempted = any(
        bool(evidence.get("environment_probe_attempted"))
        for evidence in environment_evidence
    )
    environment_executed = any(
        bool(evidence.get("environment_probe_executed"))
        for evidence in environment_evidence
    )
    external_runtime_executed = any(
        bool(evidence.get("external_runtime_executed"))
        for evidence in environment_evidence
    )
    external_assets_used = any(
        bool(evidence.get("external_assets_used")) for evidence in environment_evidence
    )
    real_simulator = any(
        bool(evidence.get("real_simulator")) for evidence in environment_evidence
    )
    gpu_executed = any(
        bool(evidence.get("gpu_executed")) for evidence in environment_evidence
    )
    environment_probe_benchmarks = [
        item["benchmark"]
        for item in results
        if item.get("evidence", {}).get("environment_probe_executed") is True
    ]
    environment_blocked_benchmarks = [
        item["benchmark"] for item in results if item.get("status") == "BLOCKED"
    ]
    return {
        "schema": SCHEMA,
        "status": overall,
        "claim": (
            "synthetic_algorithm_plus_environment_one_step_smoke"
            if algorithm_enabled and args.steps == 1
            else "synthetic_algorithm_plus_environment_short_rollout_smoke"
            if algorithm_enabled
            else "environment_one_step_smoke_only"
            if args.steps == 1
            else "environment_short_rollout_smoke_only"
        ),
        "requested_benchmarks": list(args.benchmarks),
        "parameters": {
            "steps": args.steps,
            "seed": args.seed,
            "require_cuda": args.require_cuda,
            "require_pinned_rlinf": args.require_pinned_rlinf,
            "cpu_simulator_contract": args.cpu_simulator_contract,
            "skip_algorithm_smoke": args.skip_algorithm_smoke,
        },
        "results": results,
        "private_paths_omitted_from_source": True,
        "runtime_paths_are_caller_supplied": True,
        "report_may_contain_caller_paths": True,
        "algorithm_synthetic_smoke_requested": algorithm_enabled,
        "algorithm_synthetic_smoke_executed": algorithm_executed,
        "algorithm_synthetic_smoke_completed": algorithm_completed,
        "environment_probe_attempted": environment_attempted,
        "environment_probe_executed": environment_executed,
        "external_runtime_executed": external_runtime_executed,
        "external_assets_used": external_assets_used,
        "real_simulator": real_simulator,
        "gpu_executed": gpu_executed,
        "external_openpi_executed": False,
        "environment_probe_benchmarks": environment_probe_benchmarks,
        "environment_blocked_benchmarks": environment_blocked_benchmarks,
        "short_rollout_smoke_only": args.steps > 1,
        "external_training_or_evaluation_executed": False,
        "training_or_evaluation_executed": False,
        "paper_scale": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reset and take one zero-action step in caller-supplied ManiSkill, "
            "CALVIN, and MetaWorld environments. No assets are downloaded."
        )
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        choices=("all", *BENCHMARKS),
        default=["all"],
        help="benchmark(s) to probe; default: all",
    )
    parser.add_argument(
        "--rlinf-checkout",
        "--rlinf-root",
        dest="rlinf_checkout",
        type=Path,
        help="explicit RLinf checkout containing rlinf/",
    )
    parser.add_argument(
        "--maniskill-simulator-assets",
        "--maniskill-assets",
        "--simulator-assets",
        dest="maniskill_simulator_assets",
        type=Path,
        help="explicit RLinf ManiSkill task-asset directory",
    )
    parser.add_argument(
        "--maniskill-package-assets",
        "--ms-asset-dir",
        dest="maniskill_package_assets",
        type=Path,
        help="explicit ManiSkill package asset cache (MS_ASSET_DIR)",
    )
    parser.add_argument(
        "--calvin-environment-assets",
        "--calvin-assets",
        dest="calvin_environment_assets",
        type=Path,
        help="explicit CALVIN scene/data directory",
    )
    parser.add_argument(
        "--calvin-task-oracle-annotations",
        "--calvin-annotations",
        dest="calvin_task_oracle_annotations",
        type=Path,
        help="optional explicit CALVIN task-oracle/annotation file",
    )
    parser.add_argument(
        "--metaworld-task-config",
        type=Path,
        help="MetaWorld task/prompt JSON; otherwise locate it under the supplied RLinf checkout",
    )
    parser.add_argument(
        "--metaworld-assets",
        type=Path,
        help="optional explicit MetaWorld/MuJoCo asset directory",
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="number of environment steps (default: 1)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="block before the probe when CUDA is unavailable (use for protocol qualification)",
    )
    parser.add_argument(
        "--require-pinned-rlinf",
        action="store_true",
        help=(
            "require the selected checkout's Git HEAD to match the pinned RLinf "
            f"revision {EXPECTED_RLINF_REVISION}"
        ),
    )
    parser.add_argument(
        "--cpu-simulator-contract",
        action="store_true",
        help=(
            "for ManiSkill only, replace GPU physics with an explicit CPU simulator "
            "contract; never report it as a formal benchmark run"
        ),
    )
    parser.add_argument(
        "--skip-algorithm-smoke",
        action="store_true",
        help="skip the dependency-light synthetic algorithm contract (environment-only mode)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return nonzero when any probe is blocked or fails",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="include tracebacks in failure reports"
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser


def _args_from_namespace(namespace: argparse.Namespace) -> SmokeArgs:
    requested = tuple(BENCHMARKS if "all" in namespace.benchmark else namespace.benchmark)
    # Preserve order while rejecting duplicate benchmark arguments.
    requested = tuple(dict.fromkeys(requested))
    if not requested:
        raise ValueError("at least one benchmark is required")
    if namespace.steps < 1:
        raise ValueError("--steps must be positive")
    if namespace.seed < 0:
        raise ValueError("--seed must be non-negative")
    if namespace.require_cuda and namespace.cpu_simulator_contract:
        raise ValueError("--require-cuda and --cpu-simulator-contract are mutually exclusive")
    if namespace.cpu_simulator_contract and requested != ("maniskill",):
        raise ValueError("--cpu-simulator-contract is only valid with --benchmark maniskill")
    return SmokeArgs(
        benchmarks=requested,
        rlinf_checkout=namespace.rlinf_checkout,
        maniskill_simulator_assets=namespace.maniskill_simulator_assets,
        maniskill_package_assets=namespace.maniskill_package_assets,
        calvin_environment_assets=namespace.calvin_environment_assets,
        calvin_task_oracle_annotations=namespace.calvin_task_oracle_annotations,
        metaworld_task_config=namespace.metaworld_task_config,
        metaworld_assets=namespace.metaworld_assets,
        steps=namespace.steps,
        seed=namespace.seed,
        require_cuda=namespace.require_cuda,
        require_pinned_rlinf=namespace.require_pinned_rlinf,
        cpu_simulator_contract=namespace.cpu_simulator_contract,
        skip_algorithm_smoke=namespace.skip_algorithm_smoke,
        verbose=namespace.verbose,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    namespace = parser.parse_args(argv)
    try:
        args = _args_from_namespace(namespace)
        report = run(args)
    except ValueError as error:
        parser.error(str(error))
    output = json.dumps(report, indent=2, sort_keys=True)
    if namespace.output is None:
        print(output)
    else:
        namespace.output.parent.mkdir(parents=True, exist_ok=True)
        namespace.output.write_text(output + "\n", encoding="utf-8")
        print(namespace.output)
    if report["status"] == "FAIL":
        return 1
    if namespace.strict and report["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

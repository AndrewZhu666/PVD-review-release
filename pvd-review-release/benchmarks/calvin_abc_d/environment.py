"""Runtime seam for the externally supplied CALVIN simulator.

The release does not vendor PyBullet/CALVIN or silently import the author's
machine.  ``CalvinEnvironmentAdapter`` builds the environment from the
caller-selected RLinf checkout and data directory, and exposes only the small
surface needed by training/evaluation.  Its checkpoint methods preserve the
robot controller targets as well as physics state when the upstream adapter
provides them.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .common import PROTOCOL, ContractError, activate_rlinf

_TRAINING_SCENES = ("calvin_scene_A", "calvin_scene_B", "calvin_scene_C")


def _training_scene_name(*, rank: int, local_env_index: int, local_num_envs: int) -> str:
    """Map one rank-local environment to RLinf's global ABC scene rotation.

    Fixed RLinf assigns ABC scenes with ``local_env_id % 3``.  That works when
    each worker owns several environments, but this runner deliberately owns
    one environment per distributed rank.  Include the rank offset so eight
    one-env workers cover A/B/C instead of all independently selecting A.
    """

    if rank < 0 or local_num_envs < 1 or not 0 <= local_env_index < local_num_envs:
        raise ValueError("invalid rank-local CALVIN environment index")
    global_env_index = rank * local_num_envs + local_env_index
    return _TRAINING_SCENES[global_env_index % len(_TRAINING_SCENES)]


def _repair_editable_calvin_module_file() -> None:
    """Give editable namespace installs a usable ``calvin_env.__file__``.

    Some editable-install finders expose ``calvin_env`` as a namespace package
    and leave ``__file__`` as ``None``.  The official CALVIN environment logs
    its source revision with ``Path(calvin_env.__file__)`` during construction,
    so that otherwise unrelated packaging detail prevents a simulator reset.
    Derive the package initializer from the module's explicit search path; do
    not introduce a machine-specific fallback.
    """

    try:
        module = importlib.import_module("calvin_env")
    except ImportError:
        return
    if getattr(module, "__file__", None) is not None:
        return
    for entry in getattr(module, "__path__", ()):
        root = Path(entry)
        # A normal checkout exposes ``root/__init__.py``; the official CALVIN
        # repository's editable layout commonly exposes ``root/calvin_env``
        # through a namespace finder instead.
        for candidate in (root / "__init__.py", root / "calvin_env" / "__init__.py"):
            if candidate.is_file():
                module.__file__ = str(candidate)
                return


def _build_env(
    *,
    rlinf_checkout: Path,
    environment_assets: Path,
    scene: str = PROTOCOL.evaluation_scene,
    task_suite_name: str | None = None,
    evaluation: bool = True,
    seed: int = 0,
    rank: int = 0,
    world_size: int = PROTOCOL.world_size,
    factory: Callable[..., Any] | None = None,
) -> Any:
    """Instantiate a CALVIN environment while binding its data path explicitly.

    Training and evaluation deliberately use different seams.  Training goes
    through RLinf's ``CalvinEnv`` wrapper (which owns task sampling and the
    checkpoint hooks); evaluation uses one raw scene so the official panel can
    provide explicit ``robot_obs``/``scene_obs`` reset states.  ``factory`` is
    retained as a dependency-free test seam.
    """

    activate_rlinf(rlinf_checkout)
    if factory is not None:
        try:
            return factory(
                scene=scene,
                data_path=str(environment_assets),
                task_suite_name=task_suite_name,
                evaluation=evaluation,
            )
        except TypeError:
            try:
                return factory(scene=scene, data_path=str(environment_assets))
            except TypeError:
                return factory(scene=scene)

    if not evaluation and task_suite_name == "calvin_abc":
        return _build_training_wrapper(
            rlinf_checkout=rlinf_checkout,
            environment_assets=environment_assets,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )

    return _build_raw_env(
        rlinf_checkout=rlinf_checkout,
        environment_assets=environment_assets,
        scene=scene,
    )


def _build_raw_env(
    *,
    rlinf_checkout: Path,
    environment_assets: Path,
    scene: str,
) -> Any:
    """Build one raw CALVIN scene from the selected RLinf checkout."""

    # Subprocesses may use ``spawn`` and therefore do not inherit the parent's
    # import ordering; bind the selected checkout again inside each worker.
    activate_rlinf(rlinf_checkout)
    _repair_editable_calvin_module_file()
    try:
        import hydra
        from omegaconf import OmegaConf
        from rlinf.envs.calvin import ENV_CFG_DIR
    except ImportError as error:  # pragma: no cover - exercised only in real runtime
        raise RuntimeError(
            "CALVIN runtime requires the caller's RLinf checkout plus hydra/omegaconf"
        ) from error

    config_root = Path(ENV_CFG_DIR) / ".hydra"
    merged_path = config_root / "merged_config.yaml"
    scene_path = config_root / f"{scene}.yaml"
    if not merged_path.is_file() or not scene_path.is_file():
        raise ContractError(
            "selected RLinf checkout has no CALVIN Hydra scene config; "
            f"expected {merged_path} and {scene_path}"
        )
    render_config = OmegaConf.load(merged_path)
    scene_config = OmegaConf.load(scene_path)
    # RLinf's stock ``make_env`` leaves data_path as the relative ``data``
    # default.  Bind both interpolation roots to the explicit caller asset.
    OmegaConf.set_struct(render_config, False)
    OmegaConf.set_struct(scene_config, False)
    render_config.data_path = str(environment_assets)
    scene_config.data_path = str(environment_assets)
    render_config.show_gui = False
    render_config.scene = scene_config
    render_config.env.scene_cfg = scene_config
    render_config.env.show_gui = False
    render_config.env.use_vr = False
    render_config.env.use_scene_info = True
    try:
        return hydra.utils.instantiate(
            render_config.env,
            show_gui=False,
            use_vr=False,
            use_scene_info=True,
        )
    except Exception as error:  # pragma: no cover - runtime-only dependency
        raise ContractError(
            "could not instantiate CALVIN scene with the supplied environment-assets "
            f"and scene {scene!r}: {error}"
        ) from error


def _build_training_wrapper(
    *,
    rlinf_checkout: Path,
    environment_assets: Path,
    seed: int,
    rank: int,
    world_size: int,
) -> Any:
    """Construct RLinf's ABC wrapper with a release-bound raw-env factory."""

    try:
        from rlinf.envs.calvin.calvin_gym_env import CalvinEnv
    except ImportError as error:  # pragma: no cover - runtime-only dependency
        raise RuntimeError(
            "CALVIN training requires RLinf's rlinf.envs.calvin.CalvinEnv wrapper; "
            "the selected checkout is incompatible"
        ) from error

    # The upstream class hard-codes its module-level ``make_env`` factory.
    # Override only ``get_env_fns`` so task-suite scene selection, reset
    # bookkeeping, chunk stepping, and checkpoint serialization remain owned by
    # RLinf while every subprocess receives the caller's absolute asset path.
    class BoundCalvinEnv(CalvinEnv):
        _supports_explicit_reset = False

        def get_env_fn_params(self, env_idx=None):  # type: ignore[no-untyped-def]
            selected_ids = range(self.num_envs) if env_idx is None else env_idx
            return [
                {
                    "scene": _training_scene_name(
                        rank=int(self.seed_offset),
                        local_env_index=int(local_env_id),
                        local_num_envs=int(self.num_envs),
                    )
                }
                for local_env_id in selected_ids
            ]

        def get_env_fns(self):  # type: ignore[no-untyped-def]
            env_fn_params = self.get_env_fn_params()
            env_fns = []
            for env_fn_param in env_fn_params:
                selected_scene = str(env_fn_param["scene"])

                def env_fn(
                    scene_name=selected_scene,
                    checkout=str(rlinf_checkout),
                    assets=str(environment_assets),
                    seed_offset=int(self.seed_offset),
                ):
                    os.environ["EGL_VISIBLE_DEVICES"] = str(seed_offset)
                    return _build_raw_env(
                        rlinf_checkout=Path(checkout),
                        environment_assets=Path(assets),
                        scene=scene_name,
                    )

                env_fns.append(env_fn)
            return env_fns

    cfg = SimpleNamespace(
        task_suite_name="calvin_abc",
        total_num_envs=1,
        auto_reset=False,
        ignore_terminations=False,
        use_rel_reward=True,
        reward_coef=1.0,
        max_steps_per_rollout_epoch=PROTOCOL.training_episode_limit,
        max_episode_steps=PROTOCOL.training_episode_limit,
        seed=int(seed),
        group_size=1,
        use_fixed_reset_state_ids=False,
        use_ordered_reset_state_ids=False,
        is_eval=False,
        video_cfg=SimpleNamespace(
            save_video=False,
            info_on_video=False,
            video_base_dir="",
        ),
    )
    try:
        environment = BoundCalvinEnv(cfg, 1, rank, world_size, None)
    except Exception as error:  # pragma: no cover - runtime-only dependency
        raise RuntimeError(
            f"CALVIN training could not construct RLinf's ABC environment wrapper: {error}"
        ) from error
    if not callable(getattr(environment, "capture_checkpoint_state", None)) or not callable(
        getattr(environment, "restore_checkpoint_state", None)
    ):
        raise ContractError(
            "selected RLinf CalvinEnv lacks capture_checkpoint_state/"
            "restore_checkpoint_state; exact training continuation is unavailable"
        )
    return environment


def _unwrap_reset(value: Any) -> Any:
    if isinstance(value, tuple | list) and len(value) == 2 and isinstance(value[1], Mapping):
        return value[0]
    return value


def _unwrap_step(value: Any) -> tuple[Any, Any, bool, Any]:
    if not isinstance(value, tuple | list) or len(value) not in (4, 5):
        raise RuntimeError("CALVIN environment step must return four or five values")
    if len(value) == 4:
        observation, reward, done, info = value
        return observation, reward, bool(done), info
    observation, reward, terminated, truncated, info = value
    return observation, reward, bool(terminated or truncated), info


class CalvinEnvironmentAdapter:
    """Small, testable adapter around one rank-local CALVIN environment."""

    def __init__(
        self,
        *,
        rlinf_checkout: Path,
        environment_assets: Path,
        rank: int = 0,
        world_size: int = PROTOCOL.world_size,
        seed: int = 0,
        evaluation: bool = False,
        factory: Callable[..., Any] | None = None,
    ) -> None:
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError("rank/world_size is invalid")
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.evaluation = evaluation
        self.rlinf_checkout = Path(rlinf_checkout).expanduser().resolve(strict=False)
        self.environment_assets = Path(environment_assets).expanduser().resolve(strict=False)
        if not self.environment_assets.is_dir():
            raise ContractError(f"environment-assets is not a directory: {self.environment_assets}")
        # EGL visibility is rank-local and must be set before PyBullet starts.
        os.environ.setdefault("EGL_VISIBLE_DEVICES", str(rank))
        self._factory = factory
        self.task_suite_name = "calvin_d" if evaluation else "calvin_abc"
        self.env = _build_env(
            rlinf_checkout=self.rlinf_checkout,
            environment_assets=self.environment_assets,
            scene=PROTOCOL.evaluation_scene,
            task_suite_name=self.task_suite_name,
            evaluation=evaluation,
            seed=seed,
            rank=rank,
            world_size=world_size,
            factory=factory,
        )
        if not evaluation and (
            not callable(getattr(self.env, "capture_checkpoint_state", None))
            or not callable(getattr(self.env, "restore_checkpoint_state", None))
        ):
            raise ContractError(
                "CALVIN training requires RLinf CalvinEnv checkpoint hooks; "
                "refusing a raw environment without exact continuation support"
            )
        if hasattr(self.env, "seed"):
            with contextlib.suppress(TypeError, AttributeError):
                self.env.seed(seed + rank)
        self.steps = 0

    def reset(self, *, robot_obs: Any | None = None, scene_obs: Any | None = None) -> Any:
        kwargs: dict[str, Any] = {}
        if robot_obs is not None and scene_obs is not None:
            kwargs.update(robot_obs=robot_obs, scene_obs=scene_obs)
        if kwargs and getattr(self.env, "_supports_explicit_reset", True) is False:
            raise ContractError(
                "RLinf CalvinEnv training wrapper does not accept explicit robot_obs/scene_obs; "
                "use the raw evaluation scene for Official-D panel resets"
            )
        try:
            value = self.env.reset(**kwargs)
        except TypeError as error:
            # A few CALVIN releases expose positional reset arguments only.
            if not kwargs:
                try:
                    value = self.env.reset()
                except TypeError:
                    raise error from None
            else:
                try:
                    # CALVIN's positional API is (robot_obs, scene_obs).
                    value = self.env.reset(robot_obs, scene_obs)
                except TypeError:
                    raise ContractError(
                        "CALVIN environment reset does not accept robot_obs and scene_obs"
                    ) from error
        self.steps = 0
        return _unwrap_reset(value)

    def get_obs(self) -> Any:
        if not hasattr(self.env, "get_obs"):
            raise RuntimeError("CALVIN environment does not expose get_obs")
        return self.env.get_obs()

    def get_info(self) -> Any:
        if not hasattr(self.env, "get_info"):
            raise RuntimeError("CALVIN environment does not expose get_info")
        return self.env.get_info()

    def step(self, action: Any) -> tuple[Any, Any, bool, Any]:
        observation, reward, done, info = _unwrap_step(self.env.step(action))
        self.steps += 1
        return observation, reward, done, info

    def chunk_step(self, actions: Any) -> tuple[list[Any], Any, Any, Any, list[Any]]:
        """Execute a five-action chunk, preserving RLinf's historical return shape."""

        if hasattr(self.env, "chunk_step"):
            result = self.env.chunk_step(actions)
            if not isinstance(result, tuple | list) or len(result) != 5:
                raise RuntimeError("CALVIN chunk_step must return five values")
            return tuple(result)  # type: ignore[return-value]
        # Fallback for the official single-environment wrapper.
        try:
            array = actions.detach().cpu().numpy()
        except AttributeError:
            array = actions
        if getattr(array, "ndim", 0) == 3:
            array = array[0]
        observations: list[Any] = []
        rewards: list[Any] = []
        terminations: list[bool] = []
        truncations: list[bool] = []
        infos: list[Any] = []
        for action in array:
            observation, reward, done, info = self.step(action)
            observations.append(observation)
            rewards.append(reward)
            terminations.append(done)
            truncations.append(False)
            infos.append(info)
            if done:
                break
        return observations, rewards, terminations, truncations, infos

    def capture_state(self) -> dict[str, Any]:
        """Capture continuation state, including relative-controller targets."""

        if hasattr(self.env, "capture_checkpoint_state"):
            state = self.env.capture_checkpoint_state()
        elif hasattr(self.env, "serialize"):
            state = self.env.serialize()
        else:
            raise RuntimeError(
                "CALVIN environment does not expose capture_checkpoint_state or serialize"
            )
        if not isinstance(state, Mapping):
            raise RuntimeError("CALVIN environment checkpoint must be a mapping")
        return copy.deepcopy(dict(state))

    def restore_state(self, state: Mapping[str, Any]) -> Any:
        """Restore physics and sampler metadata without selecting a new reset."""

        if hasattr(self.env, "restore_checkpoint_state"):
            return self.env.restore_checkpoint_state(copy.deepcopy(dict(state)))
        if hasattr(self.env, "restore_from_storage"):
            storage = state.get("subprocess_environment", state)
            return self.env.restore_from_storage(copy.deepcopy(storage))
        raise RuntimeError(
            "CALVIN environment does not expose restore_checkpoint_state or restore_from_storage"
        )

    def replay_state(self, state: Mapping[str, Any], action_history: list[Any]) -> Any:
        if hasattr(self.env, "replay_checkpoint_state"):
            return self.env.replay_checkpoint_state(
                copy.deepcopy(dict(state)), copy.deepcopy(action_history)
            )
        # Without the upstream replay hook there is no safe way to reconstruct
        # controller targets from only a serialized scene; fail closed.
        raise RuntimeError("CALVIN environment cannot replay a detached checkpoint")

    def restore_continuation(self, continuation: Mapping[str, Any]) -> Any:
        state = continuation.get("environment_state")
        if not isinstance(state, Mapping):
            raise ContractError("checkpoint continuation lacks environment_state")
        history = continuation.get("action_history_since_reset")
        # RLinf's current wrapper serializes controller targets and physics
        # exactly.  Prefer that hook even when history is present; replay is a
        # compatibility path only for older wrappers whose state was detached.
        if hasattr(self.env, "restore_checkpoint_state"):
            return self.restore_state(state)
        if history:
            return self.replay_state(state, list(history))
        return self.restore_state(state)

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


def policy_observation(
    observation: Mapping[str, Any], language: str | None = None
) -> dict[str, Any]:
    """Convert a raw CALVIN observation to the OpenPI/RLinf policy schema."""

    import torch

    if "main_images" in observation and "states" in observation:
        result = dict(observation)
        if language is not None:
            result["task_descriptions"] = [language]
        return result
    try:
        import numpy as np

        state = torch.from_numpy(
            np.asarray(observation["robot_obs"][:7], dtype=np.float32)
        ).unsqueeze(0)
        main = (
            torch.from_numpy(np.asarray(observation["rgb_obs"]["rgb_static"]))
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        wrist = (
            torch.from_numpy(np.asarray(observation["rgb_obs"]["rgb_gripper"]))
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("raw CALVIN observation lacks robot_obs/rgb_obs fields") from error
    return {
        "states": state,
        "main_images": main,
        "wrist_images": wrist,
        "extra_view_images": None,
        "task_descriptions": [language or ""],
    }


def prepare_actions(raw_actions: Any) -> Any:
    """Convert model actions to the seven-dimensional CALVIN command domain."""

    import numpy as np
    from rlinf.envs.action_utils import prepare_actions as host_prepare_actions

    raw = (
        raw_actions.detach().cpu().numpy()
        if hasattr(raw_actions, "detach")
        else np.asarray(raw_actions)
    )
    prepared = host_prepare_actions(
        raw_chunk_actions=raw,
        env_type="calvin",
        model_type="openpi",
        num_action_chunks=PROTOCOL.execution_prefix,
        action_dim=PROTOCOL.physical_action_dims,
    )
    if tuple(prepared.shape) != (1, PROTOCOL.execution_prefix, PROTOCOL.physical_action_dims):
        raise RuntimeError(f"prepared CALVIN action shape drift: {prepared.shape}")
    if not np.isfinite(prepared).all() or not np.isin(prepared[..., -1], (-1, 1)).all():
        raise RuntimeError("prepared CALVIN action is nonfinite or has nonbinary gripper")
    return prepared

"""Strict loader for the three paper benchmark configurations."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

PAPER_BENCHMARKS = ("calvin_abc_d", "maniskill", "metaworld_mt50")
DEFAULT_CONFIG_DIRECTORY = Path(__file__).resolve().parent / "data" / "configs"


class ConfigError(ValueError):
    """Raised when a release configuration is malformed or inconsistent."""


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    global_gradient_clip_norm: float


@dataclass(frozen=True)
class TrainingConfig:
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    world_size: int
    per_rank_batch_size: int
    global_batch_size: int
    fresh_chunks: int
    optimizer_updates: int
    environment_action_steps: int
    replay: bool
    presentations_per_chunk: int
    checkpoint_chunks: tuple[int, ...]
    rank_seed_strategy: str
    rank_seed_multiplier: int
    optimizer: OptimizerConfig


@dataclass(frozen=True)
class FlowConfig:
    solver: str
    student_steps: int
    teacher_steps: int


@dataclass(frozen=True)
class ActionConfig:
    model_horizon: int
    model_dimensions: int
    execution_prefix: int
    physical_dimensions: int


@dataclass(frozen=True)
class SupervisionConfig:
    student_path_states: str
    exclude_endpoint: bool
    detach_student_path: bool
    teacher_query: str
    time_weighting: str
    teacher_frozen: bool
    teacher_eval: bool
    teacher_gradients: bool
    teacher_actor_only: bool
    trainable_scope: str
    dagger_target: str
    dagger_loss: str


@dataclass(frozen=True)
class EvaluationConfig:
    protocol: str
    solver_steps: int
    rows_per_model: int
    episode_limit_primitive_steps: int
    primary_metric: str
    metrics: tuple[str, ...]
    panel_asset_id: str | None
    parameters: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkConfig:
    schema_version: int
    benchmark: str
    display_name: str
    task: str
    assets_manifest: str
    training: TrainingConfig
    flow: FlowConfig
    action: ActionConfig
    supervision: SupervisionConfig
    evaluation: EvaluationConfig


def _error(location: str, message: str) -> NoReturn:
    raise ConfigError(f"{location}: {message}")


def _object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _error(location, "expected a JSON object")
    if not all(isinstance(key, str) for key in value):
        _error(location, "object keys must be strings")
    return value


def _strict_fields(
    value: object,
    *,
    required: set[str],
    location: str,
) -> dict[str, object]:
    payload = _object(value, location)
    missing = sorted(required - payload.keys())
    unexpected = sorted(payload.keys() - required)
    if missing:
        _error(location, f"missing field(s): {', '.join(missing)}")
    if unexpected:
        _error(location, f"unexpected field(s): {', '.join(unexpected)}")
    return payload


def _string(value: object, location: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        _error(location, "expected a non-empty string")
    return value


def _optional_string(value: object, location: str) -> str | None:
    if value is None:
        return None
    return _string(value, location)


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        _error(location, "expected a boolean")
    return value


def _integer(value: object, location: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _error(location, f"expected an integer >= {minimum}")
    return value


def _number(value: object, location: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _error(location, "expected a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        _error(location, f"expected a finite number >= {minimum}")
    return result


def _array(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        _error(location, "expected a JSON array")
    return value


def _string_tuple(value: object, location: str) -> tuple[str, ...]:
    values = tuple(
        _string(item, f"{location}[{index}]") for index, item in enumerate(_array(value, location))
    )
    if not values:
        _error(location, "must not be empty")
    if len(values) != len(set(values)):
        _error(location, "must not contain duplicates")
    return values


def _integer_tuple(
    value: object,
    location: str,
    *,
    minimum: int = 0,
) -> tuple[int, ...]:
    values = tuple(
        _integer(item, f"{location}[{index}]", minimum=minimum)
        for index, item in enumerate(_array(value, location))
    )
    if not values:
        _error(location, "must not be empty")
    if len(values) != len(set(values)):
        _error(location, "must not contain duplicates")
    return values


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_object(path: Path) -> dict[str, object]:
    """Read a strict JSON object, rejecting duplicate keys and non-finite numbers."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda value: _error(
                    path.as_posix(), f"non-finite JSON number {value!r}"
                ),
            )
    except ConfigError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"could not read {path}: {error}") from error
    return _object(payload, path.as_posix())


def _parse_optimizer(value: object) -> OptimizerConfig:
    location = "training.optimizer"
    payload = _strict_fields(
        value,
        required={
            "name",
            "learning_rate",
            "betas",
            "epsilon",
            "weight_decay",
            "global_gradient_clip_norm",
        },
        location=location,
    )
    beta_values = _array(payload["betas"], f"{location}.betas")
    if len(beta_values) != 2:
        _error(f"{location}.betas", "expected exactly two values")
    betas = tuple(
        _number(item, f"{location}.betas[{index}]") for index, item in enumerate(beta_values)
    )
    if any(beta >= 1.0 for beta in betas):
        _error(f"{location}.betas", "values must be less than one")
    name = _string(payload["name"], f"{location}.name")
    if name != "adamw":
        _error(f"{location}.name", "paper configurations require 'adamw'")
    learning_rate = _number(payload["learning_rate"], f"{location}.learning_rate")
    epsilon = _number(payload["epsilon"], f"{location}.epsilon")
    gradient_clip = _number(
        payload["global_gradient_clip_norm"],
        f"{location}.global_gradient_clip_norm",
    )
    if learning_rate == 0.0 or epsilon == 0.0 or gradient_clip == 0.0:
        _error(location, "learning rate, epsilon, and gradient clip must be positive")
    return OptimizerConfig(
        name=name,
        learning_rate=learning_rate,
        betas=(betas[0], betas[1]),
        epsilon=epsilon,
        weight_decay=_number(payload["weight_decay"], f"{location}.weight_decay"),
        global_gradient_clip_norm=gradient_clip,
    )


def _parse_training(value: object) -> TrainingConfig:
    location = "training"
    payload = _strict_fields(
        value,
        required={
            "methods",
            "seeds",
            "world_size",
            "per_rank_batch_size",
            "global_batch_size",
            "fresh_chunks",
            "optimizer_updates",
            "environment_action_steps",
            "replay",
            "presentations_per_chunk",
            "checkpoint_chunks",
            "rank_seed_strategy",
            "rank_seed_multiplier",
            "optimizer",
        },
        location=location,
    )
    methods = _string_tuple(payload["methods"], f"{location}.methods")
    if methods != ("endpoint_dagger", "path_opd"):
        _error(
            f"{location}.methods",
            "expected ['endpoint_dagger', 'path_opd']",
        )
    seeds = _integer_tuple(payload["seeds"], f"{location}.seeds", minimum=0)
    world_size = _integer(payload["world_size"], f"{location}.world_size")
    per_rank_batch_size = _integer(
        payload["per_rank_batch_size"], f"{location}.per_rank_batch_size"
    )
    global_batch_size = _integer(payload["global_batch_size"], f"{location}.global_batch_size")
    if global_batch_size != world_size * per_rank_batch_size:
        _error(
            f"{location}.global_batch_size",
            "must equal world_size * per_rank_batch_size",
        )
    fresh_chunks = _integer(payload["fresh_chunks"], f"{location}.fresh_chunks")
    optimizer_updates = _integer(payload["optimizer_updates"], f"{location}.optimizer_updates")
    if fresh_chunks != optimizer_updates * global_batch_size:
        _error(
            location,
            "fresh_chunks must equal optimizer_updates * global_batch_size",
        )
    replay = _boolean(payload["replay"], f"{location}.replay")
    presentations = _integer(
        payload["presentations_per_chunk"],
        f"{location}.presentations_per_chunk",
    )
    if replay or presentations != 1:
        _error(location, "paper configurations require no replay and one presentation")
    checkpoints = _integer_tuple(payload["checkpoint_chunks"], f"{location}.checkpoint_chunks")
    if tuple(sorted(checkpoints)) != checkpoints or checkpoints[-1] != fresh_chunks:
        _error(
            f"{location}.checkpoint_chunks",
            "must be sorted and end at fresh_chunks",
        )
    rank_seed_strategy = _string(payload["rank_seed_strategy"], f"{location}.rank_seed_strategy")
    if rank_seed_strategy != "training_seed_plus_rank_multiple":
        _error(
            f"{location}.rank_seed_strategy",
            "expected 'training_seed_plus_rank_multiple'",
        )
    return TrainingConfig(
        methods=methods,
        seeds=seeds,
        world_size=world_size,
        per_rank_batch_size=per_rank_batch_size,
        global_batch_size=global_batch_size,
        fresh_chunks=fresh_chunks,
        optimizer_updates=optimizer_updates,
        environment_action_steps=_integer(
            payload["environment_action_steps"],
            f"{location}.environment_action_steps",
        ),
        replay=replay,
        presentations_per_chunk=presentations,
        checkpoint_chunks=checkpoints,
        rank_seed_strategy=rank_seed_strategy,
        rank_seed_multiplier=_integer(
            payload["rank_seed_multiplier"], f"{location}.rank_seed_multiplier"
        ),
        optimizer=_parse_optimizer(payload["optimizer"]),
    )


def _parse_flow(value: object) -> FlowConfig:
    location = "flow"
    payload = _strict_fields(
        value,
        required={"solver", "student_steps", "teacher_steps"},
        location=location,
    )
    solver = _string(payload["solver"], f"{location}.solver")
    if solver != "euler":
        _error(f"{location}.solver", "paper configurations require 'euler'")
    student_steps = _integer(payload["student_steps"], f"{location}.student_steps")
    teacher_steps = _integer(payload["teacher_steps"], f"{location}.teacher_steps")
    if student_steps != teacher_steps:
        _error(location, "student_steps and teacher_steps must match")
    return FlowConfig(
        solver=solver,
        student_steps=student_steps,
        teacher_steps=teacher_steps,
    )


def _parse_action(value: object) -> ActionConfig:
    location = "action"
    payload = _strict_fields(
        value,
        required={
            "model_horizon",
            "model_dimensions",
            "execution_prefix",
            "physical_dimensions",
        },
        location=location,
    )
    model_horizon = _integer(payload["model_horizon"], f"{location}.model_horizon")
    execution_prefix = _integer(payload["execution_prefix"], f"{location}.execution_prefix")
    if execution_prefix > model_horizon:
        _error(
            f"{location}.execution_prefix",
            "must not exceed model_horizon",
        )
    model_dimensions = _integer(payload["model_dimensions"], f"{location}.model_dimensions")
    physical_dimensions = _integer(
        payload["physical_dimensions"], f"{location}.physical_dimensions"
    )
    if physical_dimensions > model_dimensions:
        _error(
            f"{location}.physical_dimensions",
            "must not exceed model_dimensions",
        )
    return ActionConfig(
        model_horizon=model_horizon,
        model_dimensions=model_dimensions,
        execution_prefix=execution_prefix,
        physical_dimensions=physical_dimensions,
    )


def _parse_supervision(value: object) -> SupervisionConfig:
    location = "supervision"
    required = {
        "student_path_states",
        "exclude_endpoint",
        "detach_student_path",
        "teacher_query",
        "time_weighting",
        "teacher_frozen",
        "teacher_eval",
        "teacher_gradients",
        "teacher_actor_only",
        "trainable_scope",
        "dagger_target",
        "dagger_loss",
    }
    payload = _strict_fields(value, required=required, location=location)
    result = SupervisionConfig(
        student_path_states=_string(
            payload["student_path_states"], f"{location}.student_path_states"
        ),
        exclude_endpoint=_boolean(payload["exclude_endpoint"], f"{location}.exclude_endpoint"),
        detach_student_path=_boolean(
            payload["detach_student_path"], f"{location}.detach_student_path"
        ),
        teacher_query=_string(payload["teacher_query"], f"{location}.teacher_query"),
        time_weighting=_string(payload["time_weighting"], f"{location}.time_weighting"),
        teacher_frozen=_boolean(payload["teacher_frozen"], f"{location}.teacher_frozen"),
        teacher_eval=_boolean(payload["teacher_eval"], f"{location}.teacher_eval"),
        teacher_gradients=_boolean(payload["teacher_gradients"], f"{location}.teacher_gradients"),
        teacher_actor_only=_boolean(
            payload["teacher_actor_only"], f"{location}.teacher_actor_only"
        ),
        trainable_scope=_string(payload["trainable_scope"], f"{location}.trainable_scope"),
        dagger_target=_string(payload["dagger_target"], f"{location}.dagger_target"),
        dagger_loss=_string(payload["dagger_loss"], f"{location}.dagger_loss"),
    )
    expected: Mapping[str, object] = {
        "student_path_states": "pre_transition",
        "exclude_endpoint": True,
        "detach_student_path": True,
        "teacher_query": "same_student_states_and_times",
        "time_weighting": "uniform",
        "teacher_frozen": True,
        "teacher_eval": True,
        "teacher_gradients": False,
        "teacher_actor_only": True,
        "trainable_scope": "action_expert_only",
        "dagger_target": "teacher_rollout_endpoint",
        "dagger_loss": "native_conditional_flow_matching",
    }
    for field, expected_value in expected.items():
        if getattr(result, field) != expected_value:
            _error(f"{location}.{field}", f"expected {expected_value!r}")
    return result


def _validate_json_value(value: object, location: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _error(location, "number must be finite")
        return value
    if isinstance(value, list):
        return [
            _validate_json_value(item, f"{location}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _validate_json_value(item, f"{location}.{key}") for key, item in value.items()}
    _error(location, "value is not JSON-compatible")


def _parse_evaluation(value: object) -> EvaluationConfig:
    location = "evaluation"
    payload = _strict_fields(
        value,
        required={
            "protocol",
            "solver_steps",
            "rows_per_model",
            "episode_limit_primitive_steps",
            "primary_metric",
            "metrics",
            "panel_asset_id",
            "parameters",
        },
        location=location,
    )
    metrics = _string_tuple(payload["metrics"], f"{location}.metrics")
    primary_metric = _string(payload["primary_metric"], f"{location}.primary_metric")
    if primary_metric not in metrics:
        _error(f"{location}.primary_metric", "must also appear in metrics")
    parameters = _object(payload["parameters"], f"{location}.parameters")
    return EvaluationConfig(
        protocol=_string(payload["protocol"], f"{location}.protocol"),
        solver_steps=_integer(payload["solver_steps"], f"{location}.solver_steps"),
        rows_per_model=_integer(payload["rows_per_model"], f"{location}.rows_per_model"),
        episode_limit_primitive_steps=_integer(
            payload["episode_limit_primitive_steps"],
            f"{location}.episode_limit_primitive_steps",
        ),
        primary_metric=primary_metric,
        metrics=metrics,
        panel_asset_id=_optional_string(payload["panel_asset_id"], f"{location}.panel_asset_id"),
        parameters=_validate_json_value(parameters, f"{location}.parameters"),
    )


_EXPECTED_PAPER_CONTRACTS: dict[str, dict[str, object]] = {
    "maniskill": {
        "seeds": (0, 1, 2),
        "solver_steps": 8,
        "model_horizon": 8,
        "model_dimensions": 32,
        "execution_prefix": 5,
        "physical_dimensions": 7,
        "rank_seed_multiplier": 1,
        "evaluation_protocol": "fixed_ten_point_panel",
        "rows_per_model": 320,
    },
    "calvin_abc_d": {
        "seeds": (0, 1, 2),
        "solver_steps": 8,
        "model_horizon": 5,
        "model_dimensions": 32,
        "execution_prefix": 5,
        "physical_dimensions": 7,
        "rank_seed_multiplier": 1,
        "evaluation_protocol": "official_d",
        "rows_per_model": 1_000,
    },
    "metaworld_mt50": {
        "seeds": (0, 1),
        "solver_steps": 5,
        "model_horizon": 5,
        "model_dimensions": 32,
        "execution_prefix": 5,
        "physical_dimensions": 4,
        "rank_seed_multiplier": 100_003,
        "evaluation_protocol": "matched_mt50",
        "rows_per_model": 500,
    },
}


def _validate_named_paper_contract(config: BenchmarkConfig) -> None:
    expected = _EXPECTED_PAPER_CONTRACTS[config.benchmark]
    actual: dict[str, object] = {
        "seeds": config.training.seeds,
        "solver_steps": config.flow.student_steps,
        "model_horizon": config.action.model_horizon,
        "model_dimensions": config.action.model_dimensions,
        "execution_prefix": config.action.execution_prefix,
        "physical_dimensions": config.action.physical_dimensions,
        "rank_seed_multiplier": config.training.rank_seed_multiplier,
        "evaluation_protocol": config.evaluation.protocol,
        "rows_per_model": config.evaluation.rows_per_model,
    }
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            _error(
                f"{config.benchmark}.{field}",
                f"expected paper value {expected_value!r}, got {actual[field]!r}",
            )
    if config.training.fresh_chunks != 80_000:
        _error("training.fresh_chunks", "paper configurations require 80000")
    if config.training.optimizer_updates != 10_000:
        _error("training.optimizer_updates", "paper configurations require 10000")
    if config.training.environment_action_steps != (
        config.training.fresh_chunks * config.action.execution_prefix
    ):
        _error(
            "training.environment_action_steps",
            "must equal fresh_chunks * action.execution_prefix",
        )
    if config.evaluation.solver_steps != config.flow.student_steps:
        _error("evaluation.solver_steps", "must match flow.student_steps")


def _validate_assets_manifest_reference(config: BenchmarkConfig) -> None:
    expected = f"assets/{config.benchmark}.json"
    if config.assets_manifest != expected:
        _error("assets_manifest", f"expected {expected!r}")


def validate_benchmark_payload(value: object) -> BenchmarkConfig:
    """Validate a decoded benchmark configuration and return immutable records."""

    payload = _strict_fields(
        value,
        required={
            "schema_version",
            "benchmark",
            "display_name",
            "task",
            "assets_manifest",
            "training",
            "flow",
            "action",
            "supervision",
            "evaluation",
        },
        location="benchmark config",
    )
    schema_version = _integer(payload["schema_version"], "schema_version", minimum=1)
    if schema_version != 1:
        _error("schema_version", "only version 1 is supported")
    benchmark = _string(payload["benchmark"], "benchmark")
    if benchmark not in PAPER_BENCHMARKS:
        _error("benchmark", f"unsupported benchmark {benchmark!r}")
    assets_manifest = _string(payload["assets_manifest"], "assets_manifest")
    manifest_path = PurePosixPath(assets_manifest)
    if (
        manifest_path.is_absolute()
        or ".." in manifest_path.parts
        or manifest_path.as_posix() != assets_manifest
        or manifest_path.suffix != ".json"
    ):
        _error("assets_manifest", "must be a normalized repository-relative JSON path")
    config = BenchmarkConfig(
        schema_version=schema_version,
        benchmark=benchmark,
        display_name=_string(payload["display_name"], "display_name"),
        task=_string(payload["task"], "task"),
        assets_manifest=assets_manifest,
        training=_parse_training(payload["training"]),
        flow=_parse_flow(payload["flow"]),
        action=_parse_action(payload["action"]),
        supervision=_parse_supervision(payload["supervision"]),
        evaluation=_parse_evaluation(payload["evaluation"]),
    )
    _validate_named_paper_contract(config)
    _validate_assets_manifest_reference(config)
    return config


def load_benchmark_config(
    benchmark: str,
    *,
    config_directory: Path | str = DEFAULT_CONFIG_DIRECTORY,
) -> BenchmarkConfig:
    """Load one named paper configuration from an explicit config directory."""

    if benchmark not in PAPER_BENCHMARKS:
        raise ConfigError(f"unsupported benchmark {benchmark!r}")
    directory = Path(config_directory)
    path = directory / f"{benchmark}.json"
    config = validate_benchmark_payload(load_json_object(path))
    if config.benchmark != benchmark:
        raise ConfigError(f"{path}: benchmark field {config.benchmark!r} does not match filename")
    return config


def load_all_benchmark_configs(
    *,
    config_directory: Path | str = DEFAULT_CONFIG_DIRECTORY,
) -> dict[str, BenchmarkConfig]:
    """Load the complete, deliberately closed paper benchmark set."""

    return {
        benchmark: load_benchmark_config(
            benchmark,
            config_directory=config_directory,
        )
        for benchmark in PAPER_BENCHMARKS
    }

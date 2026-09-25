from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from path_opd.config import (  # noqa: E402
    load_all_benchmark_configs,
    load_benchmark_config,
)


def test_only_paper_benchmarks_are_configured() -> None:
    configs = load_all_benchmark_configs()

    assert set(configs) == {"calvin_abc_d", "maniskill", "metaworld_mt50"}
    assert not any(
        "libero" in path.name.casefold() for path in (REPOSITORY_ROOT / "configs").rglob("*.json")
    )


def test_shared_training_contract_matches_paper_runs() -> None:
    configs = load_all_benchmark_configs()

    for config in configs.values():
        training = config.training
        assert training.methods == ("endpoint_dagger", "path_opd")
        assert training.world_size == 8
        assert training.per_rank_batch_size == 1
        assert training.global_batch_size == 8
        assert training.fresh_chunks == 80_000
        assert training.optimizer_updates == 10_000
        assert training.environment_action_steps == 400_000
        assert training.replay is False
        assert training.presentations_per_chunk == 1
        assert training.checkpoint_chunks == (10_000, 20_000, 40_000, 60_000, 80_000)

        optimizer = training.optimizer
        assert optimizer.name == "adamw"
        assert optimizer.learning_rate == 7.91e-6
        assert optimizer.betas == (0.9, 0.95)
        assert optimizer.epsilon == 1e-5
        assert optimizer.weight_decay == 0.01
        assert optimizer.global_gradient_clip_norm == 1.0

        supervision = config.supervision
        assert supervision.student_path_states == "pre_transition"
        assert supervision.exclude_endpoint is True
        assert supervision.detach_student_path is True
        assert supervision.teacher_query == "same_student_states_and_times"
        assert supervision.time_weighting == "uniform"
        assert supervision.teacher_frozen is True
        assert supervision.teacher_eval is True
        assert supervision.teacher_gradients is False
        assert supervision.teacher_actor_only is True
        assert supervision.trainable_scope == "action_expert_only"
        assert supervision.dagger_target == "teacher_rollout_endpoint"
        assert supervision.dagger_loss == "native_conditional_flow_matching"


def test_benchmark_specific_action_and_seed_contracts() -> None:
    configs = load_all_benchmark_configs()

    expected = {
        "maniskill": {
            "seeds": (0, 1, 2),
            "solver_steps": 8,
            "model_horizon": 8,
            "model_dimensions": 32,
            "execution_prefix": 5,
            "physical_dimensions": 7,
            "rank_seed_multiplier": 1,
        },
        "calvin_abc_d": {
            "seeds": (0, 1, 2),
            "solver_steps": 8,
            "model_horizon": 5,
            "model_dimensions": 32,
            "execution_prefix": 5,
            "physical_dimensions": 7,
            "rank_seed_multiplier": 1,
        },
        "metaworld_mt50": {
            "seeds": (0, 1),
            "solver_steps": 5,
            "model_horizon": 5,
            "model_dimensions": 32,
            "execution_prefix": 5,
            "physical_dimensions": 4,
            "rank_seed_multiplier": 100_003,
        },
    }

    for benchmark, values in expected.items():
        config = configs[benchmark]
        assert config.training.seeds == values["seeds"]
        assert config.training.rank_seed_strategy == "training_seed_plus_rank_multiple"
        assert config.training.rank_seed_multiplier == values["rank_seed_multiplier"]
        assert config.flow.solver == "euler"
        assert config.flow.student_steps == values["solver_steps"]
        assert config.flow.teacher_steps == values["solver_steps"]
        assert config.action.model_horizon == values["model_horizon"]
        assert config.action.model_dimensions == values["model_dimensions"]
        assert config.action.execution_prefix == values["execution_prefix"]
        assert config.action.physical_dimensions == values["physical_dimensions"]


def test_maniskill_evaluation_contract() -> None:
    evaluation = load_benchmark_config("maniskill").evaluation

    assert evaluation.protocol == "fixed_ten_point_panel"
    assert evaluation.solver_steps == 8
    assert evaluation.rows_per_model == 320
    assert evaluation.episode_limit_primitive_steps == 80
    assert evaluation.primary_metric == "success_once"
    assert evaluation.metrics == ("success_once", "success_at_end")
    assert evaluation.panel_asset_id == "evaluation_panel"
    assert evaluation.parameters == {
        "control_frequency_hz": 5,
        "environment_count": 320,
        "fixed_reset_ids": True,
        "simulation_frequency_hz": 500,
    }


def test_calvin_evaluation_contract() -> None:
    config = load_benchmark_config("calvin_abc_d")
    evaluation = config.evaluation

    assert config.task == "ABC-D Official-D"
    assert evaluation.protocol == "official_d"
    assert evaluation.rows_per_model == 1_000
    assert evaluation.episode_limit_primitive_steps == 360
    assert evaluation.primary_metric == "average_completed_sequence_length"
    assert evaluation.panel_asset_id == "official_d_panel"
    assert evaluation.parameters == {
        "action_repeat": 8,
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 20_260_819,
        "cameras": ["rgb_static", "rgb_gripper"],
        "controller_frequency_hz": 30,
        "scene": "D",
        "sequence_length": 5,
        "sequences": 1_000,
        "simulation_frequency_hz": 240,
    }


def test_metaworld_evaluation_contract() -> None:
    evaluation = load_benchmark_config("metaworld_mt50").evaluation

    assert evaluation.protocol == "matched_mt50"
    assert evaluation.solver_steps == 5
    assert evaluation.rows_per_model == 500
    assert evaluation.episode_limit_primitive_steps == 160
    assert evaluation.primary_metric == "success_once"
    assert evaluation.panel_asset_id is None
    assert evaluation.parameters == {
        "camera_id": 2,
        "matched_variants_per_task": 10,
        "reset_settle_zero_actions": 15,
        "tasks": 50,
    }

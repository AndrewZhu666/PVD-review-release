from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.calvin_abc_d import common, environment, evaluate, train
from path_opd.config import load_benchmark_config


def test_release_config_matches_runner_protocol() -> None:
    config = load_benchmark_config("calvin_abc_d")

    assert config.evaluation.episode_limit_primitive_steps == (
        common.PROTOCOL.evaluation_subtask_limit
    )
    assert common.PROTOCOL.training_episode_limit == 480


def _panel_payload() -> dict[str, object]:
    rows = []
    for index in range(common.PROTOCOL.evaluation_rows):
        rows.append(
            {
                "sequence_index": index,
                "identity_sha256": f"{index:064x}",
                "initial_state": {"robot_obs": [0.0], "scene_obs": [0.0]},
                "subtasks": ["open_drawer"] * common.PROTOCOL.sequence_length,
            }
        )
    return {"rows": rows}


def test_custom_panel_is_structural_and_rank_assignment_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(_panel_payload()), encoding="utf-8")

    panel = common.validate_panel(path, allow_custom=True)

    assert panel["formal_panel"] is False
    assert len(panel["rows"]) == common.PROTOCOL.evaluation_rows
    for rank in range(common.PROTOCOL.world_size):
        assigned = common.rows_for_rank(panel, rank, common.PROTOCOL.world_size)
        assert len(assigned) == common.PROTOCOL.evaluation_rows // common.PROTOCOL.world_size
        assert all(row["sequence_index"] % common.PROTOCOL.world_size == rank for row in assigned)


def test_custom_panel_can_be_a_small_explicit_qualification_subset(tmp_path: Path) -> None:
    payload = _panel_payload()
    rows = payload["rows"]
    assert isinstance(rows, list)
    payload["rows"] = rows[: common.PROTOCOL.world_size]
    path = tmp_path / "subset-panel.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    panel = common.validate_panel(path, allow_custom=True)

    assert panel["formal_panel"] is False
    assert panel["row_count"] == common.PROTOCOL.world_size
    for rank in range(common.PROTOCOL.world_size):
        assigned = common.rows_for_rank(panel, rank, common.PROTOCOL.world_size)
        assert len(assigned) == 1


def test_custom_panel_subset_rejects_non_divisible_eight_rank_count(tmp_path: Path) -> None:
    payload = _panel_payload()
    rows = payload["rows"]
    assert isinstance(rows, list)
    payload["rows"] = rows[: common.PROTOCOL.world_size - 1]
    path = tmp_path / "invalid-subset-panel.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    panel = common.validate_panel(path, allow_custom=True)
    with pytest.raises(common.ContractError, match="divisible"):
        common.rows_for_rank(panel, 0, common.PROTOCOL.world_size)


def test_dry_run_does_not_import_heavy_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Dry-run is a contract/path inspection mode; external assets, including
    # the normalizer bytes, need not be mounted yet.
    normalizer = tmp_path / "norm.json"
    blocked = {"torch", "numpy", "hydra", "omegaconf", "rlinf", "calvin_agent", "pybullet"}
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object):
        if name.split(".", 1)[0] in blocked:
            raise AssertionError(f"dry-run imported heavy dependency {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = evaluate.main(
        [
            "--rlinf-checkout",
            str(tmp_path / "rlinf"),
            "--base-model",
            str(tmp_path / "base"),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--normalization-stats",
            str(normalizer),
            "--environment-assets",
            str(tmp_path / "assets"),
            "--task-oracle-annotations",
            str(tmp_path / "annotations.yaml"),
            "--panel",
            str(tmp_path / "panel.json"),
            "--output",
            str(tmp_path / "output"),
            "--method",
            "path_opd",
            "--training-seed",
            "0",
            "--dry-run",
        ]
    )

    assert result == 0


def test_reset_positional_fallback_preserves_robot_then_scene_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PositionalResetEnvironment(_FakeEnvironment):
        def reset(self, robot_obs: object, scene_obs: object, /) -> object:
            self.calls.append(("reset-positional", robot_obs, scene_obs))
            return {"observation": 0}

    fake = PositionalResetEnvironment()
    monkeypatch.setattr(environment, "activate_rlinf", lambda _checkout: None)
    assets = tmp_path / "assets"
    assets.mkdir()
    adapter = environment.CalvinEnvironmentAdapter(
        rlinf_checkout=tmp_path / "rlinf",
        environment_assets=assets,
        evaluation=True,
        factory=lambda **_: fake,
    )

    adapter.reset(robot_obs=["robot"], scene_obs=["scene"])

    assert fake.calls[0] == ("reset-positional", ["robot"], ["scene"])


def test_training_environment_rejects_non_checkpointable_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class IncompleteEnvironment:
        def reset(self) -> dict[str, int]:
            return {"observation": 0}

    assets = tmp_path / "assets"
    assets.mkdir()
    monkeypatch.setattr(environment, "activate_rlinf", lambda _checkout: None)
    with pytest.raises(common.ContractError, match="checkpoint hooks"):
        environment.CalvinEnvironmentAdapter(
            rlinf_checkout=tmp_path / "rlinf",
            environment_assets=assets,
            factory=lambda **_: IncompleteEnvironment(),
        )


def test_environment_scene_selection_defaults_to_abc_for_training_and_d_for_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeEnvironment()
    calls: list[dict[str, object]] = []

    def fake_builder(**kwargs: object) -> object:
        calls.append(kwargs)
        return fake

    monkeypatch.setattr(environment, "_build_env", fake_builder)
    assets = tmp_path / "assets"
    assets.mkdir()
    environment.CalvinEnvironmentAdapter(
        rlinf_checkout=tmp_path / "rlinf",
        environment_assets=assets,
        seed=4,
    )
    environment.CalvinEnvironmentAdapter(
        rlinf_checkout=tmp_path / "rlinf",
        environment_assets=assets,
        seed=4,
        evaluation=True,
    )

    assert calls[0]["task_suite_name"] == "calvin_abc"
    assert calls[0]["evaluation"] is False
    assert calls[1]["task_suite_name"] == "calvin_d"
    assert calls[1]["evaluation"] is True


def test_one_environment_per_rank_rotates_across_all_abc_training_scenes() -> None:
    scenes = [
        environment._training_scene_name(
            rank=rank,
            local_env_index=0,
            local_num_envs=1,
        )
        for rank in range(common.PROTOCOL.world_size)
    ]

    assert scenes == [
        "calvin_scene_A",
        "calvin_scene_B",
        "calvin_scene_C",
        "calvin_scene_A",
        "calvin_scene_B",
        "calvin_scene_C",
        "calvin_scene_A",
        "calvin_scene_B",
    ]


def test_training_chunk_progress_and_bounded_reset_policy() -> None:
    terms = [[False, False, False, False, True]]
    truncs = [[False, False, False, False, False]]
    assert train._chunk_progress(
        [[[0.0] * common.PROTOCOL.physical_action_dims] * common.PROTOCOL.execution_prefix],
        [object()] * common.PROTOCOL.execution_prefix,
        terms,
        truncs,
    ) == (common.PROTOCOL.execution_prefix, True)
    assert train._training_reset_due(
        episode_steps=common.PROTOCOL.training_episode_limit,
        chunks_since_reset=1,
        terminal=False,
    )
    assert train._training_reset_due(
        episode_steps=1,
        chunks_since_reset=common.PROTOCOL.reset_interval_chunks,
        terminal=False,
    )


def test_distributed_resume_requires_one_environment_and_rng_state_per_rank() -> None:
    continuation = {"environment_state": {"step": 1}}
    rng = {"python": (), "numpy": (), "torch": object(), "cuda": object()}
    payload = {
        "rank_states": [
            {"rank": 0, "continuation": continuation, "rng": rng},
            {"rank": 1, "continuation": {"environment_state": {"step": 2}}, "rng": rng},
        ]
    }

    selected = train._rank_state_for_resume(payload, rank=1, world_size=2)
    assert selected["continuation"]["environment_state"]["step"] == 2

    with pytest.raises(common.ContractError, match="per-rank continuation"):
        train._rank_state_for_resume(
            {"continuation": continuation},
            rank=1,
            world_size=2,
        )


def test_resume_rejects_checkpoint_bytes_that_do_not_match_manifest(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "trainer_state.pt"
    artifact.write_bytes(b"checkpoint bytes")
    expected = "0" * 64

    with pytest.raises(common.ContractError, match="file digest does not match manifest"):
        train._validate_resume_checkpoint_digest(artifact, {"checkpoint_sha256": expected})


class _FakeEnvironment:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.state = {"controller_target": [1.0], "step": 0}

    def reset(self, **kwargs: object) -> tuple[dict[str, int], dict[str, int]]:
        self.calls.append(("reset", kwargs))
        return {"observation": 0}, {"reset": 1}

    def get_obs(self) -> dict[str, int]:
        return {"observation": self.state["step"]}

    def get_info(self) -> dict[str, int]:
        return {"step": int(self.state["step"])}

    def step(self, action: object) -> tuple[dict[str, int], float, bool, dict[str, int]]:
        self.calls.append(("step", action))
        self.state["step"] += 1
        return self.get_obs(), 0.0, False, self.get_info()

    def capture_checkpoint_state(self) -> dict[str, object]:
        return dict(self.state)

    def restore_checkpoint_state(self, state: dict[str, object]) -> dict[str, object]:
        self.state = dict(state)
        return self.get_obs()

    def close(self) -> None:
        self.calls.append("close")


def test_environment_adapter_normalizes_reset_step_and_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeEnvironment()
    monkeypatch.setattr(environment, "activate_rlinf", lambda _checkout: None)
    assets = tmp_path / "assets"
    assets.mkdir()
    adapter = environment.CalvinEnvironmentAdapter(
        rlinf_checkout=tmp_path / "rlinf",
        environment_assets=assets,
        factory=lambda **_: fake,
    )

    observation = adapter.reset(robot_obs=[1], scene_obs=[2])
    stepped, reward, done, info = adapter.step([0.0] * common.PROTOCOL.physical_action_dims)
    continuation = adapter.capture_state()
    restored = adapter.restore_continuation({"environment_state": continuation})
    adapter.close()

    assert observation == {"observation": 0}
    assert stepped == {"observation": 1}
    assert reward == 0.0
    assert done is False
    assert info == {"step": 1}
    assert restored == {"observation": 1}
    assert fake.calls[0][0] == "reset"
    assert fake.calls[-1] == "close"


def test_summary_fails_closed_on_missing_duplicate_or_exception_rows() -> None:
    panel = _panel_payload()
    panel["sha256"] = "panel-digest"
    expected = panel["rows"]
    assert isinstance(expected, list)
    row = {
        "identity_sha256": expected[0]["identity_sha256"],
        "completed_prefix_length": 3,
        "exception": None,
    }

    summary = evaluate.summarize_rows(
        [row, dict(row)],
        panel=panel,
        checkpoint_sha256="checkpoint",
        normalizer_sha256="normalizer",
        method="path_opd",
        training_seed=0,
        evaluation_seed=1,
        horizon=common.PROTOCOL.evaluation_subtask_limit,
    )

    assert summary["status"] == "FAILED"
    assert summary["coverage"]["duplicate_rows"] == 1
    assert summary["coverage"]["missing_rows"] == common.PROTOCOL.evaluation_rows - 1
    assert summary["metrics"]["SR1"] == pytest.approx(1 / common.PROTOCOL.evaluation_rows)
    assert summary["formal_benchmark"] is False
    assert summary["non_formal"] is True
    assert summary["qualification"] is True


def test_resumable_rows_bind_checkpoint_panel_and_evaluation_identity(tmp_path: Path) -> None:
    path = tmp_path / "rank_000.partial.jsonl"
    row = {
        "identity_sha256": "a" * 64,
        "exception": None,
        "checkpoint_sha256": "old-checkpoint",
        "normalization_stats_sha256": "normalizer",
        "panel_sha256": "panel",
        "method": "path_opd",
        "training_seed": 0,
        "evaluation_seed": 1,
        "horizon": common.PROTOCOL.evaluation_subtask_limit,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(common.ContractError, match="resumable row identity mismatch"):
        evaluate._load_existing(
            path,
            {row["identity_sha256"]},
            expected_metadata={
                **{key: value for key, value in row.items() if key != "checkpoint_sha256"},
                "checkpoint_sha256": "new-checkpoint",
            },
        )


def test_complete_custom_or_short_calvin_summary_is_not_formal() -> None:
    panel = _panel_payload()
    panel["formal_panel"] = True
    rows = [
        {
            "identity_sha256": row["identity_sha256"],
            "completed_prefix_length": common.PROTOCOL.sequence_length,
            "exception": None,
        }
        for row in panel["rows"]
    ]

    short = evaluate.summarize_rows(
        rows,
        panel=panel,
        checkpoint_sha256="checkpoint",
        normalizer_sha256="normalizer",
        method="path_opd",
        training_seed=0,
        evaluation_seed=1,
        horizon=common.PROTOCOL.evaluation_subtask_limit - 1,
    )
    assert short["status"] == "COMPLETE"
    assert short["formal_benchmark"] is False
    assert short["non_formal"] is True
    assert short["qualification"] is True

    formal = evaluate.summarize_rows(
        rows,
        panel=panel,
        checkpoint_sha256="checkpoint",
        normalizer_sha256="normalizer",
        method="path_opd",
        training_seed=0,
        evaluation_seed=1,
        horizon=common.PROTOCOL.evaluation_subtask_limit,
    )
    assert formal["status"] == "COMPLETE"
    assert formal["formal_benchmark"] is True
    assert formal["non_formal"] is False
    assert formal["qualification"] is False


def test_checkpoint_manifest_is_required_for_real_evaluation(tmp_path: Path) -> None:
    normalizer = tmp_path / "norm.json"
    normalizer.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        world_size=common.PROTOCOL.world_size,
        evaluation_seed=0,
        horizon=common.PROTOCOL.evaluation_subtask_limit,
        device="cpu",
        rlinf_checkout=tmp_path / "rlinf",
        base_model=tmp_path / "base",
        checkpoint=tmp_path / "checkpoint.pt",
        normalization_stats=normalizer,
        environment_assets=tmp_path / "assets",
        task_oracle_annotations=tmp_path / "annotations.yaml",
        panel=tmp_path / "panel.json",
        output=tmp_path / "output",
        allow_custom_panel=True,
        dry_run=False,
    )
    args.checkpoint.write_bytes(b"checkpoint")
    args.rlinf_checkout.mkdir()
    args.base_model.mkdir()
    args.environment_assets.mkdir()
    args.task_oracle_annotations.write_text("tasks: []\n", encoding="utf-8")
    args.panel.write_text(json.dumps(_panel_payload()), encoding="utf-8")
    with pytest.raises(common.ContractError, match="manifest is missing"):
        evaluate._validate_args(args)


def test_evaluator_passes_current_code_provenance_to_release_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    normalizer = tmp_path / "norm.json"
    normalizer.write_text("{}\n", encoding="utf-8")
    environment_assets = tmp_path / "assets"
    environment_assets.mkdir()
    annotations = tmp_path / "annotations.yaml"
    annotations.write_text("tasks: []\n", encoding="utf-8")
    base_model = tmp_path / "base"
    base_model.mkdir()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps(_panel_payload()), encoding="utf-8")
    checkout = tmp_path / "rlinf"
    paths = common.RuntimePaths(
        rlinf_checkout=checkout,
        normalization_stats=normalizer,
        environment_assets=environment_assets,
        task_oracle_annotations=annotations,
        output=tmp_path / "output",
        base_model=base_model,
        checkpoint=checkpoint,
        official_d_panel=panel,
    )
    manifest = {
        "schema_version": common.SCHEMA_VERSION,
        "benchmark": common.PROTOCOL.benchmark,
        "method": "path_opd",
        "seed": 0,
        "world_size": common.PROTOCOL.world_size,
        "student_steps": common.PROTOCOL.student_steps,
        "model_horizon": common.PROTOCOL.model_horizon,
        "execution_prefix": common.PROTOCOL.execution_prefix,
        "physical_action_dims": common.PROTOCOL.physical_action_dims,
        "safe_boundary": common.SAFE_BOUNDARY,
        "release_contract": {"schema": "fixture"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    captured: dict[str, object] = {}

    def capture_release_validation(*_args: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(evaluate, "validate_evaluate_paths", lambda **_kwargs: paths)
    monkeypatch.setattr(evaluate, "validate_checkpoint_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        evaluate,
        "validate_release_checkpoint_manifest",
        capture_release_validation,
    )
    args = SimpleNamespace(
        world_size=common.PROTOCOL.world_size,
        evaluation_seed=0,
        horizon=common.PROTOCOL.evaluation_subtask_limit,
        device="cpu",
        rlinf_checkout=checkout,
        base_model=base_model,
        checkpoint=checkpoint,
        normalization_stats=normalizer,
        environment_assets=environment_assets,
        task_oracle_annotations=annotations,
        panel=panel,
        output=tmp_path / "output",
        allow_custom_panel=True,
        method="path_opd",
        training_seed=0,
        dry_run=False,
    )

    evaluate._validate_args(args)

    code = captured["code"]
    assert isinstance(code, dict)
    assert set(code) == {"release_source_sha256", "rlinf_revision"}
    assert len(code["release_source_sha256"]) == 64
    assert code["rlinf_revision"] is None


def test_both_training_arms_execute_through_released_openpi_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    class FakeForwardType:
        NFT = "NFT"
        SFT = "SFT"

    class FakeOpenPI(torch.nn.Module):
        def __init__(self, value: float, *, trainable: bool) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(value), requires_grad=trainable)

        def prepare_dagger_sft_batch(self, batch: dict[str, object]) -> dict[str, object]:
            return {"target": batch["model_action"]}

        def forward(self, **kwargs: object) -> object:
            nft_inputs = kwargs.get("nft_inputs")
            if isinstance(nft_inputs, dict):
                states = nft_inputs["x_t"]
                assert isinstance(states, torch.Tensor)
                return {"v_theta": states * 0 + self.anchor}
            data = kwargs["data"]
            assert isinstance(data, dict)
            target = data["target"]
            assert isinstance(target, torch.Tensor)
            return target.square().mean() * self.anchor

    monkeypatch.setattr(train._PathRuntime, "_forward_enum", staticmethod(lambda: FakeForwardType))
    chains = torch.zeros(
        1,
        common.PROTOCOL.student_steps + 1,
        common.PROTOCOL.model_horizon,
        common.PROTOCOL.model_action_dims,
        requires_grad=True,
    )
    forward_inputs = {
        "chains": chains,
        "observation/image": torch.zeros(1, 1),
        "observation/state": torch.zeros(1, 4),
        "tokenized_prompt": torch.zeros(1, 2, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    student_result = {"forward_inputs": forward_inputs}
    student = FakeOpenPI(1.0, trainable=True)
    teacher = FakeOpenPI(0.0, trainable=False)

    path_loss = train._PathRuntime(teacher).loss(student, student_result)
    torch.testing.assert_close(path_loss, torch.tensor(1.0))
    path_loss.backward()
    torch.testing.assert_close(student.anchor.grad, torch.tensor(2.0))
    assert teacher.anchor.grad is None
    assert chains.grad is None

    student.anchor.grad = None
    teacher_result = {
        "model_actions": torch.ones(
            1,
            common.PROTOCOL.model_horizon,
            common.PROTOCOL.model_action_dims,
        )
    }
    dagger_loss = train._dagger_loss(student, student_result, teacher_result)
    torch.testing.assert_close(dagger_loss, torch.tensor(1.0))
    dagger_loss.backward()
    torch.testing.assert_close(student.anchor.grad, torch.tensor(1.0))

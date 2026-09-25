from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import pytest

from path_opd.release_contract import build_checkpoint_manifest, tree_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIRECTORY = REPOSITORY_ROOT / "benchmarks" / "metaworld_mt50"
sys.path.insert(0, str(RUNNER_DIRECTORY))

import common as metaworld_common  # noqa: E402
import evaluate as metaworld_evaluate  # noqa: E402
import train as metaworld_train  # noqa: E402


def _fake_release_inputs(
    tmp_path: Path, *, task_config_layout: str = "metaworld"
) -> dict[str, Path]:
    checkout = tmp_path / "RLinf"
    (checkout / "rlinf/models/embodiment/openpi").mkdir(parents=True)
    (checkout / "rlinf/models/embodiment/base_policy.py").write_text(
        "# contract fixture\n", encoding="utf-8"
    )
    (checkout / "rlinf/models/embodiment/openpi/__init__.py").write_text(
        "# contract fixture\n", encoding="utf-8"
    )
    if task_config_layout not in {"metaworld", "sim"}:
        raise ValueError(f"unsupported task config layout: {task_config_layout}")
    config = checkout / (
        "rlinf/envs/metaworld/metaworld_config.json"
        if task_config_layout == "metaworld"
        else "rlinf/envs/sim/metaworld/metaworld_config.json"
    )
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "TASK_DESCRIPTIONS": {
                    f"task-{index}-v3": f"perform task {index}"
                    for index in range(metaworld_common.PROTOCOL.task_count)
                }
            }
        ),
        encoding="utf-8",
    )
    base = tmp_path / "base"
    teacher = tmp_path / "teacher"
    checkpoint = tmp_path / "checkpoint"
    for model in (base, teacher, checkpoint):
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"fixture")
    norm = tmp_path / "norm_stats.json"
    norm.write_text("{}\n", encoding="utf-8")
    return {
        "checkout": checkout,
        "task_config": config,
        "base": base,
        "teacher": teacher,
        "checkpoint": checkpoint,
        "norm": norm,
    }


@pytest.mark.parametrize("task_config_layout", ("metaworld", "sim"))
def test_dry_run_path_validation_resolves_both_task_config_layouts(
    tmp_path: Path, task_config_layout: str
) -> None:
    paths = _fake_release_inputs(tmp_path, task_config_layout=task_config_layout)
    expected = paths["task_config"].resolve()

    train_paths = metaworld_common.validate_train_paths(
        rlinf_checkout=paths["checkout"],
        base_model=paths["base"],
        teacher_model=paths["teacher"],
        norm_stats=paths["norm"],
        output=tmp_path / "train-output",
        require_existing=False,
    )
    evaluate_paths = metaworld_common.validate_evaluate_paths(
        rlinf_checkout=paths["checkout"],
        checkpoint=paths["checkpoint"],
        norm_stats=paths["norm"],
        output=tmp_path / "evaluation-output",
        require_existing=False,
    )

    assert train_paths.task_config == expected
    assert evaluate_paths.task_config == expected
    assert len(metaworld_common.validate_task_config(expected)) == 50


def _write_checkpoint_sidecar(
    paths: dict[str, Path], *, method: str = "path_opd", seed: int = 0
) -> None:
    """Add the release manifest emitted by train.py to the tiny fixture."""

    release = build_checkpoint_manifest(
        benchmark=metaworld_common.PROTOCOL.suite,
        method=method,
        seed=seed,
        world_size=metaworld_common.PROTOCOL.world_size,
        update=1,
        contract=metaworld_common.paper_contract(),
        assets={
            "base_model": "a" * 64,
            "frozen_teacher": "b" * 64,
            "normalization_stats": tree_sha256(paths["norm"]),
            "rlinf_task_config": tree_sha256(
                paths["checkout"] / "rlinf/envs/metaworld/metaworld_config.json"
            ),
        },
        code={
            "release_source_sha256": tree_sha256(
                REPOSITORY_ROOT, include=("src", "benchmarks", "configs")
            ),
            "rlinf_revision": None,
        },
        extra={"target_chunks": metaworld_common.PROTOCOL.target_chunks},
    )
    checkpoint_weight = paths["checkpoint"] / "model.safetensors"
    sidecar = {
        "schema_version": 1,
        "method": method,
        "seed": seed,
        "world_size": metaworld_common.PROTOCOL.world_size,
        "config_name": metaworld_common.PROTOCOL.config_name,
        "model_horizon": metaworld_common.PROTOCOL.model_horizon,
        "execution_prefix": metaworld_common.PROTOCOL.execution_prefix,
        "physical_action_dims": metaworld_common.PROTOCOL.physical_action_dims,
        "model_action_dims": metaworld_common.PROTOCOL.model_action_dims,
        "student_steps": metaworld_common.PROTOCOL.student_steps,
        "teacher_steps": metaworld_common.PROTOCOL.teacher_steps,
        "episode_limit": metaworld_common.PROTOCOL.episode_limit,
        "optimizer_update": 1,
        "normalizer_sha256": metaworld_common.file_sha256(paths["norm"]),
        "model_file_sha256": metaworld_common.file_sha256(checkpoint_weight),
        "release_contract": release,
    }
    (paths["checkpoint"] / "manifest.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )


def _block_heavy_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__
    blocked = {"gymnasium", "metaworld", "omegaconf", "openpi", "torch"}

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.split(".", 1)[0] in blocked:
            raise AssertionError(f"dry-run imported heavy dependency {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_train_dry_run_is_cpu_only_and_reports_fixed_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _fake_release_inputs(tmp_path)
    output = tmp_path / "train-output"
    _block_heavy_imports(monkeypatch)

    result = metaworld_train.main(
        [
            "--method",
            "path_opd",
            "--rlinf-checkout",
            str(paths["checkout"]),
            "--base",
            str(paths["base"]),
            "--teacher",
            str(paths["teacher"]),
            "--norm",
            str(paths["norm"]),
            "--output",
            str(output),
            "--qualification",
            "--dry-run",
        ]
    )

    assert result == 0
    assert not output.exists()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "DRY_RUN_ONLY"
    assert report["execution_performed"] is False
    assert report["training"] == {
        "effective_chunks": 8,
        "effective_updates": 1,
        "launcher": "torchrun",
        "method": "path_opd",
        "planned_updates": 10_000,
        "run_scope": "qualification",
        "seed": 0,
        "target_chunks": 80_000,
        "backend": "nccl",
    }
    protocol = report["protocol"]
    assert protocol["world_size"] == 8
    assert protocol["student_steps"] == 5
    assert protocol["model_horizon"] == 5
    assert protocol["execution_prefix"] == 5
    assert protocol["physical_action_dims"] == 4


def test_train_dry_run_is_declaration_only_without_external_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _block_heavy_imports(monkeypatch)
    result = metaworld_train.main(
        [
            "--method",
            "path_opd",
            "--rlinf-checkout",
            str(tmp_path / "missing-checkout"),
            "--base",
            str(tmp_path / "missing-base"),
            "--teacher",
            str(tmp_path / "missing-teacher"),
            "--norm",
            str(tmp_path / "missing-norm.json"),
            "--output",
            str(tmp_path / "output"),
            "--dry-run",
        ]
    )
    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "DRY_RUN_ONLY"
    assert report["execution_performed"] is False
    assert report["checks"]["paths_exist"] is False
    assert report["checks"]["task_config_validated"] is False


def test_documented_dry_run_outputs_are_copyable_from_checkout() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert "--output artifacts/metaworld-dry-run/train" in readme
    assert "--output artifacts/metaworld-dry-run/evaluate" in readme
    assert "--output /path/to/new_run" not in readme
    assert "--output /path/to/new_eval" not in readme


def test_real_training_still_rejects_missing_external_files(tmp_path: Path) -> None:
    result = metaworld_train.main(
        [
            "--method",
            "path_opd",
            "--rlinf-checkout",
            str(tmp_path / "missing-checkout"),
            "--base",
            str(tmp_path / "missing-base"),
            "--teacher",
            str(tmp_path / "missing-teacher"),
            "--norm",
            str(tmp_path / "missing-norm.json"),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    assert result == 2


def test_training_rejects_seed_outside_the_published_two_seed_protocol(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        metaworld_train.build_parser().parse_args(
            [
                "--method",
                "path_opd",
                "--rlinf-checkout",
                str(tmp_path / "checkout"),
                "--base",
                str(tmp_path / "base"),
                "--teacher",
                str(tmp_path / "teacher"),
                "--norm",
                str(tmp_path / "norm.json"),
                "--output",
                str(tmp_path / "output"),
                "--seed",
                "2",
            ]
        )


def test_resume_target_is_an_absolute_update_count() -> None:
    metaworld_train._validate_resume_target(2, 1)
    metaworld_train._validate_resume_target(None, 1)

    with pytest.raises(ValueError, match="absolute target"):
        metaworld_train._validate_resume_target(1, 1)
    with pytest.raises(ValueError, match="optimizer_update"):
        metaworld_train._validate_resume_target(2, None)
    with pytest.raises(ValueError, match="optimizer_update"):
        metaworld_train._validate_resume_target(2, True)


def test_evaluate_dry_run_declares_reconstruction_and_exact_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _fake_release_inputs(tmp_path)
    output = tmp_path / "evaluation-output"
    _block_heavy_imports(monkeypatch)

    result = metaworld_evaluate.main(
        [
            "--rlinf-checkout",
            str(paths["checkout"]),
            "--checkpoint",
            str(paths["checkpoint"]),
            "--norm",
            str(paths["norm"]),
            "--output",
            str(output),
            "--method",
            "endpoint_dagger",
            "--training-seed",
            "1",
            "--dry-run",
        ]
    )

    assert result == 0
    assert not output.exists()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "DRY_RUN_ONLY"
    assert report["evaluation"]["expected_rows"] == 500
    assert report["evaluation"]["protocol"]["episode_limit_primitive_steps"] == 160
    provenance = report["evaluation"]["provenance"]
    assert provenance["implementation"] == "reconstructed_evaluator"
    assert provenance["byte_identical_to_historical_evaluator"] is False


def test_evaluate_dry_run_is_declaration_only_without_external_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _block_heavy_imports(monkeypatch)
    result = metaworld_evaluate.main(
        [
            "--rlinf-checkout",
            str(tmp_path / "missing-checkout"),
            "--checkpoint",
            str(tmp_path / "missing-checkpoint"),
            "--norm",
            str(tmp_path / "missing-norm.json"),
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
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "DRY_RUN_ONLY"
    assert report["execution_performed"] is False
    assert report["checks"]["paths_exist"] is False
    assert report["evaluation"]["panel"]["verified"] is False


def test_real_evaluation_requires_checkpoint_release_manifest(tmp_path: Path) -> None:
    paths = _fake_release_inputs(tmp_path)
    args = metaworld_evaluate.build_parser().parse_args(
        [
            "--rlinf-checkout",
            str(paths["checkout"]),
            "--checkpoint",
            str(paths["checkpoint"]),
            "--norm",
            str(paths["norm"]),
            "--output",
            str(tmp_path / "evaluation-output"),
            "--method",
            "path_opd",
            "--training-seed",
            "0",
            "--device",
            "cpu",
        ]
    )
    with pytest.raises(ValueError, match="manifest"):
        metaworld_evaluate._validate_args(args)


def test_real_evaluation_validates_checkpoint_method_and_seed(tmp_path: Path) -> None:
    paths = _fake_release_inputs(tmp_path)
    _write_checkpoint_sidecar(paths)
    args = metaworld_evaluate.build_parser().parse_args(
        [
            "--rlinf-checkout",
            str(paths["checkout"]),
            "--checkpoint",
            str(paths["checkpoint"]),
            "--norm",
            str(paths["norm"]),
            "--output",
            str(tmp_path / "evaluation-output"),
            "--method",
            "path_opd",
            "--training-seed",
            "0",
            "--device",
            "cpu",
        ]
    )
    metaworld_evaluate._validate_args(args)
    assert args.checkpoint_contract["verified"] is True

    args.method = "endpoint_dagger"
    with pytest.raises(ValueError, match="method"):
        metaworld_evaluate._validate_args(args)


def test_real_evaluation_rejects_checkpoint_from_different_release_source(
    tmp_path: Path,
) -> None:
    paths = _fake_release_inputs(tmp_path)
    _write_checkpoint_sidecar(paths)
    sidecar_path = paths["checkpoint"] / "manifest.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["release_contract"]["code"]["release_source_sha256"] = "d" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    args = metaworld_evaluate.build_parser().parse_args(
        [
            "--rlinf-checkout",
            str(paths["checkout"]),
            "--checkpoint",
            str(paths["checkpoint"]),
            "--norm",
            str(paths["norm"]),
            "--output",
            str(tmp_path / "evaluation-output"),
            "--method",
            "path_opd",
            "--training-seed",
            "0",
            "--device",
            "cpu",
        ]
    )

    with pytest.raises(ValueError, match="release contract mismatch"):
        metaworld_evaluate._validate_args(args)


def test_custom_panel_digest_binds_task_config_provenance() -> None:
    tasks = [("task-0-v3", "pick up the object")]
    first = metaworld_evaluate.generated_panel_sha256(
        tasks, task_config_sha256="a" * 64
    )
    second = metaworld_evaluate.generated_panel_sha256(
        tasks, task_config_sha256="b" * 64
    )
    assert first != second
    payload = metaworld_evaluate.panel_identity_payload(
        tasks, task_config_sha256="a" * 64
    )
    assert payload["schema"] == "path-opd-metaworld-custom-panel-v1"
    assert payload["source"]["task_config_sha256"] == "a" * 64


def test_completed_rows_reject_a_changed_generated_task_variant() -> None:
    row = {"task_variant_sha256": "a" * 64}

    metaworld_evaluate._validate_completed_task_variant(
        row, task_index=4, trial_id=2, actual_sha256="a" * 64
    )
    with pytest.raises(ValueError, match="task variant mismatch"):
        metaworld_evaluate._validate_completed_task_variant(
            row, task_index=4, trial_id=2, actual_sha256="b" * 64
        )


def test_complete_resume_revalidates_every_generated_task_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_items = [(f"task-{index}", "prompt") for index in range(2)]
    expected_rows = {
        (task_index, trial_id): {
            "task_variant_sha256": metaworld_evaluate.hashlib.sha256(
                f"{task_index}:{trial_id}".encode()
            ).hexdigest()
        }
        for task_index in range(len(task_items))
        for trial_id in range(metaworld_common.PROTOCOL.variants_per_task)
    }

    class FakeTask:
        def __init__(self, data: bytes) -> None:
            self.data = data

    class FakeSelector:
        def __init__(self, task_index: int) -> None:
            self.tasks = [
                FakeTask(f"{task_index}:{trial_id}".encode())
                for trial_id in range(metaworld_common.PROTOCOL.variants_per_task)
            ]
            self.disabled = False

        def toggle_sample_tasks_on_reset(self, enabled: bool) -> None:
            self.disabled = not enabled

    class FakeEnvironment:
        def __init__(self, task_index: int) -> None:
            self.selector = FakeSelector(task_index)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeGym:
        def __init__(self) -> None:
            self.environments: list[FakeEnvironment] = []

        def make(self, _name: str, *, env_name: str, **_kwargs: object) -> FakeEnvironment:
            env = FakeEnvironment(int(env_name.split("-")[-1]))
            self.environments.append(env)
            return env

    fake_gym = FakeGym()
    monkeypatch.setattr(
        metaworld_evaluate,
        "_find_task_selector",
        lambda env: env.selector,
    )
    metaworld_evaluate._validate_all_completed_task_variants(
        fake_gym,
        task_items,
        evaluation_seed_value=195,
        completed_rows=expected_rows,
    )

    assert len(fake_gym.environments) == len(task_items)
    assert all(env.selector.disabled and env.closed for env in fake_gym.environments)


def test_matched_seed_functions_do_not_depend_on_method_or_checkpoint() -> None:
    assert metaworld_evaluate.row_seed(195, 0, 0) == 195
    assert metaworld_evaluate.row_seed(195, 49, 9) == 694
    assert metaworld_evaluate.inference_seed(195, 3, 7, 2) == 1_950_307_002


def test_summary_requires_all_500_unique_successful_rows() -> None:
    rows = []
    for task_index in range(metaworld_common.PROTOCOL.task_count):
        for trial_id in range(metaworld_common.PROTOCOL.variants_per_task):
            rows.append(
                {
                    "task_index": task_index,
                    "task": f"task-{task_index}-v3",
                    "trial_id": trial_id,
                    "success_once": trial_id % 2 == 0,
                    "exception": None,
                }
            )

    summary = metaworld_evaluate.summarize_rows(
        rows,
        checkpoint_sha256="a" * 64,
        normalizer_sha256="c" * 64,
        method="path_opd",
        training_seed=0,
        evaluation_seed_value=195,
    )

    assert summary["status"] == "COMPLETE"
    assert summary["formal_benchmark"] is False
    assert summary["non_formal"] is True
    assert summary["evaluator_equivalence"] is False
    assert summary["qualification"] is True
    assert summary["evaluated_rows"] == 500
    assert summary["successes"] == 250
    assert summary["success_once_rate"] == 0.5
    assert all(item["evaluated"] == 10 for item in summary["per_task"])


def test_summary_uses_latest_retry_and_marks_remaining_exception() -> None:
    failed = {
        "task_index": 0,
        "task": "task-0-v3",
        "trial_id": 0,
        "success_once": False,
        "exception": "RuntimeError: first attempt",
    }
    recovered = {**failed, "success_once": True, "exception": None}
    still_failed = {**failed, "trial_id": 1}

    summary = metaworld_evaluate.summarize_rows(
        [failed, recovered, still_failed],
        checkpoint_sha256="b" * 64,
        normalizer_sha256="d" * 64,
        method="endpoint_dagger",
        training_seed=1,
        evaluation_seed_value=195,
    )

    assert summary["status"] == "FAILED"
    assert summary["records_in_jsonl"] == 3
    assert summary["unique_rows"] == 2
    assert summary["evaluated_rows"] == 1
    assert summary["exception_rows"] == 1
    assert summary["successes"] == 1


def test_output_cannot_overlap_source_or_model_inputs(tmp_path: Path) -> None:
    paths = _fake_release_inputs(tmp_path)

    with pytest.raises(ValueError, match="outside input"):
        metaworld_common.validate_train_paths(
            rlinf_checkout=paths["checkout"],
            base_model=paths["base"],
            teacher_model=paths["teacher"],
            norm_stats=paths["norm"],
            output=paths["base"] / "run",
        )


def test_release_runner_calls_real_shared_objectives() -> None:
    source = (RUNNER_DIRECTORY / "train.py").read_text(encoding="utf-8")

    assert "PathOPD(" in source
    assert "objective.supervise(" in source
    assert "OpenPIAdapter.from_rollout(" in source
    assert "adapter.endpoint_dagger_loss(" in source
    assert "model.predict_action_batch(" in source


def test_both_training_arms_execute_through_released_openpi_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    import path_opd.core as path_core

    class FakeForwardType:
        NFT = "NFT"
        SFT = "SFT"

    class FakeOpenPI(torch.nn.Module):
        def __init__(self, value: float, *, trainable: bool) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(value), requires_grad=trainable)
            self.context_seen: list[float] = []

        def prepare_dagger_sft_batch(self, batch: dict[str, object]) -> dict[str, object]:
            return {"target": batch["model_action"]}

        def forward(self, **kwargs: object) -> object:
            nft_inputs = kwargs.get("nft_inputs")
            if isinstance(nft_inputs, dict):
                context = kwargs["forward_inputs"]
                assert isinstance(context, dict)
                state = context["observation/state"]
                assert isinstance(state, torch.Tensor)
                self.context_seen.append(float(state[0, 0]))
                states = nft_inputs["x_t"]
                assert isinstance(states, torch.Tensor)
                return {"v_theta": states * 0 + self.anchor}
            data = kwargs["data"]
            assert isinstance(data, dict)
            target = data["target"]
            assert isinstance(target, torch.Tensor)
            return target.square().mean() * self.anchor

    monkeypatch.setattr(metaworld_train, "_forward_type", lambda: FakeForwardType)
    chains = torch.zeros(
        1,
        metaworld_common.PROTOCOL.student_steps + 1,
        metaworld_common.PROTOCOL.model_horizon,
        metaworld_common.PROTOCOL.model_action_dims,
        requires_grad=True,
    )
    forward_inputs = {
        "chains": chains,
        "observation/image": torch.zeros(1, 1),
        "observation/state": torch.zeros(1, 4),
        "tokenized_prompt": torch.zeros(1, 2, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(1, 2, dtype=torch.bool),
        "action": torch.zeros(1, 20),
    }
    student_result = {"forward_inputs": forward_inputs}
    student = FakeOpenPI(1.0, trainable=True)
    teacher = FakeOpenPI(0.0, trainable=False)

    fingerprint_calls = 0
    original_fingerprint = path_core._module_fingerprint

    def count_fingerprint(model: torch.nn.Module) -> str:
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        return original_fingerprint(model)

    monkeypatch.setattr(path_core, "_module_fingerprint", count_fingerprint)
    runtime = metaworld_train._PathLossRuntime(teacher)

    path_loss = metaworld_train._path_loss(student, student_result, runtime)
    torch.testing.assert_close(path_loss, torch.tensor(1.0))
    path_loss.backward()
    torch.testing.assert_close(student.anchor.grad, torch.tensor(2.0))
    assert teacher.anchor.grad is None
    assert chains.grad is None

    student.anchor.grad = None
    second_chains = chains.detach().clone().requires_grad_(True)
    second_inputs = dict(forward_inputs)
    second_inputs["chains"] = second_chains
    second_inputs["observation/state"] = torch.full((1, 4), 2.0)
    second_result = {"forward_inputs": second_inputs}
    second_loss = metaworld_train._path_loss(student, second_result, runtime)
    torch.testing.assert_close(second_loss, torch.tensor(1.0))
    second_loss.backward()
    torch.testing.assert_close(student.anchor.grad, torch.tensor(2.0))
    assert fingerprint_calls == 0
    assert teacher.context_seen == [0.0, 2.0]
    assert teacher.anchor.grad is None
    assert second_chains.grad is None

    student.anchor.grad = None
    teacher_result = {
        "model_actions": torch.ones(
            1,
            metaworld_common.PROTOCOL.model_horizon,
            metaworld_common.PROTOCOL.model_action_dims,
        )
    }
    dagger_loss = metaworld_train._dagger_loss(student, student_result, teacher_result)
    torch.testing.assert_close(dagger_loss, torch.tensor(1.0))
    dagger_loss.backward()
    torch.testing.assert_close(student.anchor.grad, torch.tensor(1.0))


def test_release_runner_contains_no_private_machine_defaults() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in RUNNER_DIRECTORY.iterdir()
        if path.suffix in {".py", ".md"}
    )

    forbidden = (
        "/" + "home/",
        "/" + "data/users/",
        "example-host-123",
        "REPO" + " = Path(",
    )
    assert not any(value in sources for value in forbidden)

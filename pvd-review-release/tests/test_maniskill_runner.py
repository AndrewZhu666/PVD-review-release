from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest
import torch

from benchmarks.maniskill import common as maniskill_common
from benchmarks.maniskill import evaluate as maniskill_evaluate
from benchmarks.maniskill import train as maniskill_train

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIRECTORY = REPOSITORY_ROOT / "benchmarks" / "maniskill"
HEX_DIGEST = "a" * 64


class _MappingOmegaConf:
    @staticmethod
    def create(value: dict[str, object]) -> dict[str, object]:
        return value


def _train_arguments(tmp_path: Path, method: str = "path_opd") -> list[str]:
    return [
        "--method",
        method,
        "--rlinf-root",
        str(tmp_path / "RLinf"),
        "--base-model",
        str(tmp_path / "base"),
        "--teacher-model",
        str(tmp_path / "teacher"),
        "--normalization-stats",
        str(tmp_path / "norm.json"),
        "--simulator-assets",
        str(tmp_path / "simulator-assets"),
        "--maniskill-assets",
        str(tmp_path / "maniskill-assets"),
        "--output-dir",
        str(tmp_path / "output"),
    ]


def _evaluation_arguments(tmp_path: Path) -> list[str]:
    return [
        "--rlinf-root",
        str(tmp_path / "RLinf"),
        "--base-model",
        str(tmp_path / "base"),
        "--normalization-stats",
        str(tmp_path / "norm.json"),
        "--simulator-assets",
        str(tmp_path / "simulator-assets"),
        "--maniskill-assets",
        str(tmp_path / "maniskill-assets"),
        "--checkpoint",
        str(tmp_path / "checkpoint.pt"),
        "--checkpoint-sha256",
        HEX_DIGEST,
        "--method",
        "path_opd",
        "--training-seed",
        "0",
        "--panel",
        str(tmp_path / "panel.json"),
        "--panel-sha256",
        HEX_DIGEST,
        "--output",
        str(tmp_path / "results.json"),
    ]


def _block_heavy_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__
    blocked = {"mani_skill", "numpy", "omegaconf", "openpi", "rlinf", "torch"}

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


@pytest.mark.parametrize("method", ["path_opd", "endpoint_dagger"])
def test_train_dry_run_is_dependency_free_and_reports_qualification(
    method: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _block_heavy_imports(monkeypatch)

    result = maniskill_train.main(
        [*_train_arguments(tmp_path, method), "--max-updates", "1", "--dry-run"]
    )

    assert result == 0
    assert not (tmp_path / "output").exists()
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "path-opd-maniskill-train-v1"
    assert report["status"] == "DRY_RUN"
    assert report["executes_benchmark"] is False
    assert report["resolved"] == {
        "evaluation_rows": 320,
        "executed_prefix": 5,
        "formal_full_budget": False,
        "global_batch_size": 8,
        "global_fresh_chunks": 8,
        "local_fresh_chunks": 1,
        "max_updates": 1,
        "method": method,
        "model_horizon": 8,
        "physical_action_dims": 7,
        "qualification": True,
        "seed": 0,
        "solver_steps": 8,
        "world_size": 8,
    }
    assert report["runtime"]["dry_run_performs_training"] is False


def test_default_training_contract_is_full_80k_chunk_budget(tmp_path: Path) -> None:
    args = maniskill_train.build_parser().parse_args(_train_arguments(tmp_path))
    report = maniskill_train._resolved_report(args, require_existing=False)

    assert report["resolved"]["max_updates"] == 10_000
    assert report["resolved"]["global_fresh_chunks"] == 80_000
    assert report["resolved"]["formal_full_budget"] is True
    assert report["resolved"]["qualification"] is False


def test_teacher_normalizer_defaults_to_student_for_legacy_cli(tmp_path: Path) -> None:
    args = maniskill_train.build_parser().parse_args(_train_arguments(tmp_path))

    assert args.teacher_normalization_stats is None
    assert maniskill_train.teacher_normalization_stats(args) == args.normalization_stats
    assets = maniskill_train._asset_arguments(args)
    assert assets["teacher_normalization_stats"] == args.normalization_stats


def test_teacher_normalizer_override_reaches_model_and_asset_contract(
    tmp_path: Path,
) -> None:
    teacher_norm = tmp_path / "teacher-norm.json"
    arguments = [
        *_train_arguments(tmp_path),
        "--teacher-normalization-stats",
        str(teacher_norm),
    ]
    args = maniskill_train.build_parser().parse_args(arguments)

    assert maniskill_train.teacher_normalization_stats(args) == teacher_norm
    assets = maniskill_train._asset_arguments(args)
    assert assets["normalization_stats"] != assets["teacher_normalization_stats"]
    config = maniskill_train._model_config(
        _MappingOmegaConf,
        model_path=args.teacher_model,
        normalization_stats=maniskill_train.teacher_normalization_stats(args),
        train_expert_only=False,
    )
    assert config["openpi_data"]["norm_stats_path"] == str(teacher_norm)


def test_old_resume_manifest_is_accepted_only_without_teacher_override(
    tmp_path: Path,
) -> None:
    args = maniskill_train.build_parser().parse_args(_train_arguments(tmp_path))
    release_assets = {
        "base_model": "a" * 64,
        "frozen_teacher": "b" * 64,
        "normalization_stats": "c" * 64,
        "teacher_normalization_stats": "d" * 64,
        "simulator_assets": "e" * 64,
        "maniskill_package_assets": "f" * 64,
    }
    old_manifest = {
        "asset_sha256": {
            key: value
            for key, value in release_assets.items()
            if key != "teacher_normalization_stats"
        }
    }

    compatible = maniskill_train._resume_asset_hashes(args, release_assets, old_manifest)
    assert "teacher_normalization_stats" not in compatible

    args = maniskill_train.build_parser().parse_args(
        [
            *_train_arguments(tmp_path),
            "--teacher-normalization-stats",
            str(tmp_path / "teacher.json"),
        ]
    )
    strict = maniskill_train._resume_asset_hashes(args, release_assets, old_manifest)
    assert strict["teacher_normalization_stats"] == "d" * 64


def test_new_checkpoint_identity_binds_both_normalizers(tmp_path: Path) -> None:
    arguments = [
        *_train_arguments(tmp_path),
        "--teacher-normalization-stats",
        str(tmp_path / "teacher-norm.json"),
    ]
    args = maniskill_train.build_parser().parse_args(arguments)
    for path, contents in (
        (args.base_model / "model.safetensors", b"base"),
        (args.teacher_model / "model.safetensors", b"teacher"),
        (args.normalization_stats, b"student norm"),
        (args.teacher_normalization_stats, b"teacher norm"),
        (args.simulator_assets / "asset", b"simulator"),
        (args.maniskill_package_assets / "asset", b"package"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    hashes = maniskill_train._release_asset_hashes(args)

    assert set(hashes) == set(maniskill_common.ASSET_PARAMETERS_TRAIN)
    assert hashes["normalization_stats"] != hashes["teacher_normalization_stats"]


def test_evaluator_accepts_checkpoint_with_extra_teacher_normalizer_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = maniskill_evaluate.build_parser().parse_args(_evaluation_arguments(tmp_path))
    evaluator_assets = {
        "base_model": "1" * 64,
        "normalization_stats": "2" * 64,
        "simulator_assets": "3" * 64,
        "maniskill_package_assets": "4" * 64,
    }
    code = {
        "release_source_sha256": "5" * 64,
        "rlinf_revision": "fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c",
    }
    release = maniskill_train.build_checkpoint_manifest(
        benchmark="maniskill",
        method="path_opd",
        seed=0,
        world_size=maniskill_common.CONTRACT.world_size,
        update=1,
        contract=maniskill_common.paper_contract_dict(),
        assets={
            **evaluator_assets,
            "frozen_teacher": "6" * 64,
            "teacher_normalization_stats": "7" * 64,
        },
        code=code,
    )
    manifest = {
        "checkpoint_sha256": HEX_DIGEST,
        "method": "path_opd",
        "seed": 0,
        "world_size": maniskill_common.CONTRACT.world_size,
        "contract": maniskill_common.resolve_contract(
            method="path_opd",
            seed=0,
            world_size=maniskill_common.CONTRACT.world_size,
            max_updates=1,
        ).as_dict(),
        "release_contract": release,
    }
    monkeypatch.setattr(maniskill_evaluate, "_load_checkpoint_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        maniskill_evaluate,
        "_checkpoint_asset_hashes",
        lambda _args: evaluator_assets,
    )
    monkeypatch.setattr(maniskill_evaluate, "tree_sha256", lambda *_args, **_kwargs: "5" * 64)
    monkeypatch.setattr(
        maniskill_evaluate,
        "git_revision",
        lambda _path: "fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c",
    )

    report = maniskill_evaluate._validate_checkpoint_contract(
        args,
        checkpoint_sha256=HEX_DIGEST,
    )

    assert report["verified"] is True
    assert "teacher_normalization_stats" not in report["asset_hashes_verified"]


def test_training_and_evaluation_share_explicit_control_mode() -> None:
    train_config = maniskill_train._environment_config(_MappingOmegaConf)
    evaluation_config = maniskill_evaluate._environment_config(_MappingOmegaConf, 40)

    expected = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
    assert expected == maniskill_common.MANISKILL_CONTROL_MODE
    assert train_config["init_params"]["control_mode"] == expected
    assert evaluation_config["init_params"]["control_mode"] == expected
    assert maniskill_common.paper_contract_dict()["action_contract"]["control_mode"] == expected


def test_teacher_download_root_is_rejected_and_actor_directory_is_accepted(
    tmp_path: Path,
) -> None:
    base_model = tmp_path / "base"
    base_model.mkdir()
    (base_model / "model.safetensors").write_bytes(b"base")
    teacher_root = tmp_path / "teacher-download"
    teacher_actor = teacher_root / "actor"
    teacher_actor.mkdir(parents=True)
    (teacher_actor / "model.safetensors").write_bytes(b"teacher")
    normalization_stats = tmp_path / "norm.json"
    normalization_stats.write_text("{}\n", encoding="utf-8")
    simulator_assets = tmp_path / "simulator-assets"
    simulator_assets.mkdir()
    package_assets = tmp_path / "package-assets"
    package_assets.mkdir()
    supplied = {
        "base_model": base_model,
        "frozen_teacher": teacher_root,
        "normalization_stats": normalization_stats,
        "simulator_assets": simulator_assets,
        "maniskill_package_assets": package_assets,
    }

    with pytest.raises(maniskill_common.ContractError) as error:
        maniskill_common.validate_assets(supplied, purpose="train", require_existing=True)
    assert "weights are under actor/model.safetensors" in str(error.value)
    assert f"Pass {teacher_actor} as --teacher-model" in str(error.value)

    supplied["frozen_teacher"] = teacher_actor
    resolved = maniskill_common.validate_assets(
        supplied,
        purpose="train",
        require_existing=True,
    )
    assert resolved["frozen_teacher"] == teacher_actor


def test_evaluate_dry_run_declares_exact_320_row_schema_without_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _block_heavy_imports(monkeypatch)

    result = maniskill_evaluate.main([*_evaluation_arguments(tmp_path), "--dry-run"])

    assert result == 0
    assert not (tmp_path / "results.json").exists()
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "path-opd-maniskill-evaluation-v1"
    assert report["status"] == "DRY_RUN"
    assert report["executes_benchmark"] is False
    assert report["resolved"]["fixed_rows"] == 320
    assert report["resolved"]["episode_primitive_steps"] == 80
    assert report["output_schema"]["coverage"]["expected_rows"] == "integer"
    assert set(report["output_schema"]["metrics"]) == {
        "success_at_end",
        "success_once",
    }
    assert report["runtime"]["dry_run_performs_evaluation"] is False


def test_cli_requires_explicit_assets_and_valid_hashes(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        maniskill_train.build_parser().parse_args(
            ["--method", "path_opd", "--output-dir", str(tmp_path / "output")]
        )

    invalid = _evaluation_arguments(tmp_path)
    invalid[invalid.index(HEX_DIGEST)] = "not-a-sha256"
    with pytest.raises(SystemExit):
        maniskill_evaluate.build_parser().parse_args(invalid)


def test_evaluation_rejects_non_recorded_world_size(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = maniskill_evaluate.main(
        [*_evaluation_arguments(tmp_path), "--world-size", "4", "--dry-run"]
    )

    assert result == 2
    assert "requires world_size=8" in capsys.readouterr().err


def test_evaluation_rejects_checkpoint_without_release_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _block_heavy_imports(monkeypatch)
    arguments = _evaluation_arguments(tmp_path)
    for relative in maniskill_common.RLINF_REQUIRED_FILES:
        required = tmp_path / "RLinf" / relative
        required.parent.mkdir(parents=True, exist_ok=True)
        required.write_text("# fixture\n", encoding="utf-8")
    (tmp_path / "base").mkdir()
    (tmp_path / "norm.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "simulator-assets").mkdir()
    (tmp_path / "maniskill-assets").mkdir()
    (tmp_path / "checkpoint.pt").write_bytes(b"not-a-checkpoint")
    (tmp_path / "panel.json").write_text("{}\n", encoding="utf-8")
    arguments[arguments.index(HEX_DIGEST)] = maniskill_common.sha256_file(
        tmp_path / "checkpoint.pt"
    )

    result = maniskill_evaluate.main([*arguments])

    assert result == 2
    assert "checkpoint manifest.json is required" in capsys.readouterr().err


def _panel_payload() -> dict[str, object]:
    rows = []
    for rank in range(maniskill_common.CONTRACT.world_size):
        for slot in range(
            maniskill_common.CONTRACT.evaluation_rows // maniskill_common.CONTRACT.world_size
        ):
            reset_id = rank * 10_000 + slot
            rows.append(
                {
                    "reset_episode_id": reset_id,
                    "initial_state_id": reset_id,
                    "worker_rank": rank,
                    "pipeline_stage": 0,
                    "env_slot": slot,
                }
            )
    return {
        "environment_id": maniskill_common.CONTRACT.task,
        "denominator": maniskill_common.CONTRACT.evaluation_rows,
        "rows": rows,
    }


def _write_panel(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_panel_validator_accepts_exact_rank_ownership(tmp_path: Path) -> None:
    panel = maniskill_common.validate_panel(_write_panel(tmp_path, _panel_payload()))

    assert panel["row_count"] == 320
    assert len(maniskill_common.rows_for_rank(panel, 3, 8)) == 40
    assert maniskill_common.rows_for_rank(panel, 3, 8)[0]["env_slot"] == 0
    assert len(panel["ordered_reset_ids_sha256"]) == 64


@pytest.mark.parametrize("fault", ["row_count", "duplicate_id", "ownership"])
def test_panel_validator_rejects_incomplete_or_ambiguous_panels(
    fault: str,
    tmp_path: Path,
) -> None:
    payload = _panel_payload()
    rows = payload["rows"]
    assert isinstance(rows, list)
    if fault == "row_count":
        rows.pop()
    elif fault == "duplicate_id":
        rows[1]["reset_episode_id"] = rows[0]["reset_episode_id"]
        rows[1]["initial_state_id"] = rows[0]["initial_state_id"]
    else:
        rows[-1]["env_slot"] = 0

    with pytest.raises(maniskill_common.ContractError):
        maniskill_common.validate_panel(_write_panel(tmp_path, payload))


def test_release_runner_calls_real_rlinf_openpi_and_path_opd_apis() -> None:
    train_source = (RUNNER_DIRECTORY / "train.py").read_text(encoding="utf-8")
    evaluate_source = (RUNNER_DIRECTORY / "evaluate.py").read_text(encoding="utf-8")

    for fragment in (
        "from path_opd.core import (",
        "objective.supervise(",
        "OpenPIAdapter.from_rollout(",
        "adapter.endpoint_dagger_loss(",
        "student.predict_action_batch(",
        "from rlinf.envs.maniskill.maniskill_offload_env import _ManiskillEnvCore",
        "total_num_processes=args.world_size",
    ):
        assert fragment in train_source
    for fragment in (
        "from rlinf.envs.maniskill.maniskill_env import ManiskillEnv",
        "model.predict_action_batch(",
        "prepare_actions(",
        "env.chunk_step(",
    ):
        assert fragment in evaluate_source


def test_inference_rollout_context_is_cloned_before_autograd_reuse() -> None:
    with torch.inference_mode():
        rollout = {
            "chains": torch.ones(1, 2, 1, 1),
            "nested": {"action": torch.zeros(1, 1)},
        }

    cloned = maniskill_train._tree_detach_clone(rollout, torch)

    assert cloned["chains"].is_inference() is False
    assert cloned["nested"]["action"].is_inference() is False
    assert cloned["chains"] is not rollout["chains"]


def test_environment_observations_are_recursively_moved_to_policy_device() -> None:
    observation = {
        "image": torch.ones(1, 3),
        "nested": [torch.zeros(1), (torch.arange(2), "prompt")],
    }

    moved = maniskill_train._tree_to_device(observation, torch.device("meta"), torch)

    assert moved["image"].device.type == "meta"
    assert moved["nested"][0].device.type == "meta"
    assert moved["nested"][1][0].device.type == "meta"
    assert moved["nested"][1][1] == "prompt"


def test_evaluator_observations_are_recursively_moved_to_policy_device() -> None:
    observation = {
        "image": torch.ones(1, 3),
        "nested": [torch.zeros(1), (torch.arange(2), "prompt")],
    }

    moved = maniskill_evaluate._tree_to_device(
        observation, torch.device("meta"), torch
    )

    assert moved["image"].device.type == "meta"
    assert moved["nested"][0].device.type == "meta"
    assert moved["nested"][1][0].device.type == "meta"
    assert moved["nested"][1][1] == "prompt"


def test_release_runner_contains_no_private_machine_defaults() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in RUNNER_DIRECTORY.iterdir()
        if path.suffix == ".py"
    )

    private_path_prefixes = (
        "/" + "home" + "/",
        "/" + "data" + "/" + "users" + "/",
    )
    assert all(prefix not in sources for prefix in private_path_prefixes)
    assert "example-host-123" not in sources

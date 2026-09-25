from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import benchmark_environment_smoke as smoke


def _args(**overrides: object) -> smoke.SmokeArgs:
    values: dict[str, object] = {
        "benchmarks": ("maniskill",),
        "rlinf_checkout": None,
        "maniskill_simulator_assets": None,
        "maniskill_package_assets": None,
        "calvin_environment_assets": None,
        "calvin_task_oracle_annotations": None,
        "metaworld_task_config": None,
        "metaworld_assets": None,
        "steps": 1,
        "seed": 0,
        "require_cuda": False,
        "require_pinned_rlinf": False,
        "cpu_simulator_contract": False,
        "skip_algorithm_smoke": True,
        "verbose": False,
    }
    values.update(overrides)
    return smoke.SmokeArgs(**values)


def _fake_checkout(tmp_path: Path, benchmark: str) -> Path:
    checkout = tmp_path / "RLinf"
    required = smoke._required_files(checkout, benchmark)
    for relative in required:
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(
                json.dumps(
                    {
                        "TASK_DESCRIPTIONS": {
                            f"task-{index}-v3": f"prompt {index}" for index in range(50)
                        }
                    }
                ),
                encoding="utf-8",
            )
        else:
            path.write_text("# fixture\n", encoding="utf-8")
    (checkout / "rlinf").mkdir(exist_ok=True)
    return checkout


def test_cli_without_paths_reports_blocked_without_guessing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = smoke.main(["--benchmark", "all"])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == smoke.SCHEMA
    assert payload["status"] == "BLOCKED"
    assert payload["training_or_evaluation_executed"] is False
    assert payload["algorithm_synthetic_smoke_executed"] is True
    assert payload["claim"] == "synthetic_algorithm_plus_environment_one_step_smoke"
    assert {item["status"] for item in payload["results"]} == {"BLOCKED"}
    assert all(item["checks"]["rlinf_checkout"]["supplied"] is False for item in payload["results"])
    assert all(
        item["checks"]["algorithm_synthetic_contract"]["status"] == "PASS"
        for item in payload["results"]
    )
    assert all(
        item["evidence"]["environment_probe_executed"] is False
        and item["evidence"]["external_runtime_executed"] is False
        and item["evidence"]["one_step_smoke_only"] is True
        and item["evidence"]["short_rollout_smoke_only"] is False
        and item["evidence"]["formal_benchmark"] is False
        for item in payload["results"]
    )
    assert payload["short_rollout_smoke_only"] is False


def test_strict_mode_returns_nonzero_for_blocked(capsys: pytest.CaptureFixture[str]) -> None:
    result = smoke.main(["--benchmark", "metaworld_mt50", "--strict"])

    assert result == 2
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"


def test_parser_aliases_and_cpu_contract_flag() -> None:
    namespace = smoke._parser().parse_args(
        [
            "--benchmark",
            "maniskill",
            "--rlinf-root",
            "/tmp/rlinf",
            "--maniskill-assets",
            "/tmp/simulator",
            "--ms-asset-dir",
            "/tmp/package",
            "--cpu-simulator-contract",
        ]
    )
    args = smoke._args_from_namespace(namespace)

    assert args.rlinf_checkout == Path("/tmp/rlinf")
    assert args.maniskill_simulator_assets == Path("/tmp/simulator")
    assert args.maniskill_package_assets == Path("/tmp/package")
    assert args.cpu_simulator_contract is True
    assert args.skip_algorithm_smoke is False

    pinned = smoke._args_from_namespace(
        smoke._parser().parse_args(["--benchmark", "maniskill", "--require-pinned-rlinf"])
    )
    assert pinned.require_pinned_rlinf is True

    with pytest.raises(ValueError, match="mutually exclusive"):
        smoke._args_from_namespace(
            smoke._parser().parse_args(
                ["--benchmark", "maniskill", "--require-cuda", "--cpu-simulator-contract"]
            )
        )
    with pytest.raises(ValueError, match="only valid with --benchmark maniskill"):
        smoke._args_from_namespace(
            smoke._parser().parse_args(
                ["--benchmark", "all", "--cpu-simulator-contract"]
            )
        )


def test_metaworld_headless_rendering_selects_egl_without_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    selected = smoke._prepare_metaworld_headless_rendering()

    assert selected == "egl:auto_egl_no_display"
    assert smoke.os.environ["MUJOCO_GL"] == "egl"
    assert smoke.os.environ["PYOPENGL_PLATFORM"] == "egl"


def test_metaworld_headless_rendering_preserves_caller_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    monkeypatch.delenv("DISPLAY", raising=False)

    selected = smoke._prepare_metaworld_headless_rendering()

    assert selected == "osmesa:caller"
    assert smoke.os.environ["MUJOCO_GL"] == "osmesa"
    assert smoke.os.environ["PYOPENGL_PLATFORM"] == "osmesa"


def test_graphics_context_failure_is_blocked_not_runner_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_checkout(tmp_path, "metaworld_mt50")
    monkeypatch.setattr(smoke, "_dependency_check", lambda _benchmark, _checkout: [])
    monkeypatch.setitem(
        smoke._RUNNERS,
        "metaworld_mt50",
        lambda _args: (_ for _ in ()).throw(
            RuntimeError("an OpenGL platform library has not been loaded into this process")
        ),
    )

    report = smoke.run(
        _args(
            benchmarks=("metaworld_mt50",),
            rlinf_checkout=checkout,
            skip_algorithm_smoke=True,
        )
    )

    assert report["status"] == "BLOCKED"
    result = report["results"][0]
    assert result["status"] == "BLOCKED"
    assert "headless graphics context" in result["blockers"][0]


def test_wrapped_dependency_failure_is_blocked_not_runner_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_checkout(tmp_path, "calvin_abc_d")
    monkeypatch.setattr(smoke, "_dependency_check", lambda _benchmark, _checkout: [])

    def runner(_args: smoke.SmokeArgs) -> dict[str, object]:
        try:
            raise ModuleNotFoundError("No module named 'pytorch_lightning'")
        except ModuleNotFoundError as cause:
            raise RuntimeError("CALVIN agent initialization failed") from cause

    monkeypatch.setitem(smoke._RUNNERS, "calvin_abc_d", runner)
    report = smoke.run(
        _args(
            benchmarks=("calvin_abc_d",),
            rlinf_checkout=checkout,
            calvin_environment_assets=tmp_path,
        )
    )

    assert report["status"] == "BLOCKED"
    result = report["results"][0]
    assert result["status"] == "BLOCKED"
    assert "runtime dependency unavailable" in result["blockers"][0]


def test_missing_simulator_asset_failure_is_blocked_not_runner_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_checkout(tmp_path, "calvin_abc_d")
    monkeypatch.setattr(smoke, "_dependency_check", lambda _benchmark, _checkout: [])
    monkeypatch.setitem(
        smoke._RUNNERS,
        "calvin_abc_d",
        lambda _args: (_ for _ in ()).throw(
            RuntimeError("pybullet.error: Cannot load URDF file.")
        ),
    )

    report = smoke.run(
        _args(
            benchmarks=("calvin_abc_d",),
            rlinf_checkout=checkout,
            calvin_environment_assets=tmp_path,
        )
    )

    assert report["status"] == "BLOCKED"
    result = report["results"][0]
    assert result["status"] == "BLOCKED"
    assert "simulator asset bundle is incomplete" in result["blockers"][0]


def test_preflight_blocks_before_dependency_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("dependency discovery should not run before path preflight")

    monkeypatch.setattr(smoke.importlib.util, "find_spec", forbidden)
    checks, blockers = smoke._preflight(_args(), "maniskill")

    assert blockers
    assert checks["rlinf_checkout"]["supplied"] is False
    assert called is False


def test_metaworld_task_config_is_derived_only_from_selected_checkout(tmp_path: Path) -> None:
    checkout = _fake_checkout(tmp_path, "metaworld_mt50")
    checks, blockers = smoke._preflight(
        _args(benchmarks=("metaworld_mt50",), rlinf_checkout=checkout),
        "metaworld_mt50",
    )

    assert not any("task-config" in blocker for blocker in blockers)
    assert checks["metaworld_task_config"]["exists"] is True
    assert checks["metaworld_task_config"]["path"].endswith("metaworld_config.json")


def test_pinned_rlinf_requirement_fails_closed_without_git_metadata(tmp_path: Path) -> None:
    checkout = _fake_checkout(tmp_path, "metaworld_mt50")
    checks, blockers = smoke._preflight(
        _args(
            benchmarks=("metaworld_mt50",),
            rlinf_checkout=checkout,
            require_pinned_rlinf=True,
        ),
        "metaworld_mt50",
    )

    assert checks["rlinf_revision"]["verified"] is False
    assert any("revision could not be verified" in blocker for blocker in blockers)


def test_dependency_block_is_reported_without_running_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_checkout(tmp_path, "metaworld_mt50")
    monkeypatch.setattr(smoke, "_dependency_check", lambda _benchmark, _checkout: ["fake_env"])
    runner_called = False

    def forbidden(_args: smoke.SmokeArgs) -> dict[str, object]:
        nonlocal runner_called
        runner_called = True
        raise AssertionError("runner must not execute when a dependency is missing")

    monkeypatch.setitem(smoke._RUNNERS, "metaworld_mt50", forbidden)
    report = smoke.run(
        _args(benchmarks=("metaworld_mt50",), rlinf_checkout=checkout)
    )

    assert report["status"] == "BLOCKED"
    result = report["results"][0]
    assert result["checks"]["dependencies"]["missing"] == ["fake_env"]
    assert runner_called is False


def test_runner_pass_is_explicitly_environment_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_checkout(tmp_path, "metaworld_mt50")
    monkeypatch.setattr(smoke, "_dependency_check", lambda _benchmark, _checkout: [])
    monkeypatch.setitem(
        smoke._RUNNERS,
        "metaworld_mt50",
        lambda _args: {"steps_completed": 1, "formal_benchmark": False},
    )
    report = smoke.run(
        _args(benchmarks=("metaworld_mt50",), rlinf_checkout=checkout)
    )

    assert report["status"] == "PASS"
    assert report["claim"] == "environment_one_step_smoke_only"
    assert report["training_or_evaluation_executed"] is False
    assert report["results"][0]["evidence"]["formal_benchmark"] is False


def test_blocked_environment_retains_algorithm_pass_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithm_report = {
        "status": "PASS",
        "claim": "synthetic_contract_smoke_only",
        "synthetic_only": True,
        "contract": {"physical_dimensions": 4},
        "checks": {"exact_resume": True},
        "exact_resume": True,
    }
    monkeypatch.setattr(
        smoke, "_run_algorithm_contract", lambda _args, _benchmark: algorithm_report
    )

    report = smoke.run(
        _args(benchmarks=("metaworld_mt50",), skip_algorithm_smoke=False)
    )

    assert report["status"] == "BLOCKED"
    result = report["results"][0]
    assert result["status"] == "BLOCKED"
    assert result["checks"]["algorithm_synthetic_contract"] == algorithm_report
    assert result["evidence"]["algorithm_synthetic_contract"] == algorithm_report


def test_algorithm_contract_failure_is_not_downgraded_to_environment_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_args: smoke.SmokeArgs, _benchmark: str) -> dict[str, object]:
        raise RuntimeError("contract drift")

    monkeypatch.setattr(smoke, "_run_algorithm_contract", fail)
    preflight_called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal preflight_called
        preflight_called = True
        raise AssertionError("environment preflight must not hide an algorithm failure")

    monkeypatch.setattr(smoke, "_preflight", forbidden)
    report = smoke.run(
        _args(benchmarks=("metaworld_mt50",), skip_algorithm_smoke=False)
    )

    assert report["status"] == "FAIL"
    assert report["results"][0]["status"] == "FAIL"
    assert "contract drift" in report["results"][0]["blockers"][0]
    assert preflight_called is False


def test_skip_algorithm_smoke_is_explicit_environment_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_checkout(tmp_path, "metaworld_mt50")
    monkeypatch.setattr(smoke, "_dependency_check", lambda _benchmark, _checkout: [])
    monkeypatch.setitem(
        smoke._RUNNERS,
        "metaworld_mt50",
        lambda _args: {"steps_completed": 1, "formal_benchmark": False},
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("algorithm smoke should have been skipped")

    monkeypatch.setattr(smoke, "_run_algorithm_contract", forbidden)
    report = smoke.run(
        _args(
            benchmarks=("metaworld_mt50",),
            rlinf_checkout=checkout,
            skip_algorithm_smoke=True,
        )
    )

    assert report["status"] == "PASS"
    assert report["claim"] == "environment_one_step_smoke_only"
    assert report["algorithm_synthetic_smoke_executed"] is False
    assert report["environment_probe_executed"] is True
    assert report["external_runtime_executed"] is True
    assert report["external_assets_used"] is True
    assert report["real_simulator"] is True
    assert report["gpu_executed"] is False
    assert report["external_training_or_evaluation_executed"] is False
    assert (
        report["results"][0]["checks"]["algorithm_synthetic_contract"]["status"]
        == "SKIPPED"
    )


def test_multiple_steps_are_labelled_as_short_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_checkout(tmp_path, "metaworld_mt50")
    monkeypatch.setattr(smoke, "_dependency_check", lambda _benchmark, _checkout: [])
    monkeypatch.setitem(
        smoke._RUNNERS,
        "metaworld_mt50",
        lambda _args: {"steps_completed": 2, "formal_benchmark": False},
    )
    report = smoke.run(
        _args(
            benchmarks=("metaworld_mt50",),
            rlinf_checkout=checkout,
            steps=2,
        )
    )

    assert report["status"] == "PASS"
    assert report["claim"] == "environment_short_rollout_smoke_only"
    evidence = report["results"][0]["evidence"]
    assert evidence["requested_steps"] == 2
    assert evidence["one_step_smoke_only"] is False
    assert evidence["short_rollout_smoke_only"] is True
    assert report["short_rollout_smoke_only"] is True

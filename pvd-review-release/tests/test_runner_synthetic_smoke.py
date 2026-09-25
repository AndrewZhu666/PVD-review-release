from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.calvin_abc_d import evaluate as calvin_evaluate
from benchmarks.calvin_abc_d import train as calvin_train
from benchmarks.maniskill import evaluate as maniskill_evaluate
from benchmarks.maniskill import train as maniskill_train
from benchmarks.metaworld_mt50 import evaluate as metaworld_evaluate
from benchmarks.metaworld_mt50 import train as metaworld_train


@pytest.mark.parametrize(
    ("runner", "benchmark", "entrypoint"),
    [
        (maniskill_train, "maniskill", "benchmarks/maniskill/train.py"),
        (calvin_train, "calvin_abc_d", "benchmarks/calvin_abc_d/train.py"),
        (metaworld_train, "metaworld_mt50", "benchmarks/metaworld_mt50/train.py"),
    ],
)
def test_runner_synthetic_smoke_is_end_to_end_and_asset_free(
    runner: object,
    benchmark: str,
    entrypoint: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "report.json"
    work_dir = tmp_path / "work"
    result = runner.main(
        [
            "--synthetic-smoke",
            "--work-dir",
            str(work_dir),
            "--output",
            str(output),
            "--updates",
            "2",
            "--batch-size",
            "1",
            "--seed",
            "13",
        ]
    )

    assert result == 0
    assert capsys.readouterr().out.strip() == str(output)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "path-opd-runner-synthetic-smoke-v1"
    assert report["status"] == "PASS"
    assert report["claim"] == "synthetic_runner_contract_smoke_only"
    assert report["synthetic_only"] is True
    assert report["runner_entrypoint_executed"] is True
    assert report["entrypoint"] == entrypoint
    assert report["external_runtime_executed"] is False
    assert report["external_assets_used"] is False
    assert report["external_openpi_executed"] is False
    assert report["real_simulator"] is False
    assert report["gpu_executed"] is False
    assert report["paper_scale"] is False
    benchmark_report = report["benchmark_report"]
    assert benchmark_report["benchmark"] == benchmark
    assert benchmark_report["status"] == "PASS"
    assert benchmark_report["synthetic_only"] is True
    assert benchmark_report["checks"]["exact_resume"] is True
    assert benchmark_report["checks"]["endpoint_dagger_target_detached"] is True
    assert benchmark_report["checks"]["physical_action_contract"] is True


def test_synthetic_smoke_does_not_accept_real_runner_arguments(tmp_path: Path) -> None:
    """Synthetic mode remains an explicit alternate command, not a bypass."""

    with pytest.raises(SystemExit):
        maniskill_train.main(
            [
                "--synthetic-smoke",
                "--method",
                "path_opd",
                "--work-dir",
                str(tmp_path),
            ]
        )


@pytest.mark.parametrize(
    ("evaluator", "benchmark", "entrypoint"),
    [
        (maniskill_evaluate, "maniskill", "benchmarks/maniskill/evaluate.py"),
        (calvin_evaluate, "calvin_abc_d", "benchmarks/calvin_abc_d/evaluate.py"),
        (metaworld_evaluate, "metaworld_mt50", "benchmarks/metaworld_mt50/evaluate.py"),
    ],
)
def test_evaluator_synthetic_smoke_is_end_to_end_and_asset_free(
    evaluator: object,
    benchmark: str,
    entrypoint: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "evaluator-report.json"
    result = evaluator.main(
        [
            "--synthetic-smoke",
            "--work-dir",
            str(tmp_path / "work"),
            "--output",
            str(output),
            "--updates",
            "2",
            "--batch-size",
            "1",
            "--seed",
            "13",
        ]
    )

    assert result == 0
    assert capsys.readouterr().out.strip() == str(output)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "path-opd-evaluator-synthetic-smoke-v1"
    assert report["status"] == "PASS"
    assert report["claim"] == "synthetic_evaluator_contract_smoke_only"
    assert report["synthetic_only"] is True
    assert report["evaluator_entrypoint_executed"] is True
    assert report["training_smoke_executed"] is True
    assert report["entrypoint"] == entrypoint
    assert report["external_runtime_executed"] is False
    assert report["external_assets_used"] is False
    assert report["external_openpi_executed"] is False
    assert report["real_simulator"] is False
    assert report["gpu_executed"] is False
    assert report["paper_scale"] is False
    assert report["checkpoint"]["schema"] == (
        "path-opd-benchmark-synthetic-checkpoint-v1"
    )
    assert report["panel"]["formal"] is False
    assert report["panel"]["rows"] > 0
    assert report["panel"]["rows"] == report["panel"]["unique_identities"]
    assert report["action_trace"]["panel_rows"] == report["panel"]["rows"]
    assert report["action_trace"]["unique_panel_identities"] == report["panel"]["unique_identities"]
    assert (
        report["action_trace"]["ordered_identity_sha256"]
        == report["panel"]["ordered_identity_sha256"]
    )
    assert report["action_trace"]["trace"]["bitwise_endpoint"] is True
    assert report["action_trace"]["trace"]["bitwise_executed_action"] is True
    assert all(report["checks"].values())
    assert report["training_report"]["benchmark"] == benchmark


def test_evaluator_synthetic_smoke_rejects_real_arguments(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        maniskill_evaluate.main(
            [
                "--synthetic-smoke",
                "--method",
                "path_opd",
                "--work-dir",
                str(tmp_path),
            ]
        )


def test_metaworld_evaluator_synthetic_import_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing optional runtime must produce a controlled CLI status."""

    from path_opd.adapters import benchmark_smoke

    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise ImportError("synthetic smoke dependency unavailable")

    monkeypatch.setattr(benchmark_smoke, "run_evaluator_synthetic_smoke", unavailable)
    result = metaworld_evaluate.main(
        [
            "--synthetic-smoke",
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )

    assert result == 2
    assert "synthetic smoke dependency unavailable" in capsys.readouterr().err

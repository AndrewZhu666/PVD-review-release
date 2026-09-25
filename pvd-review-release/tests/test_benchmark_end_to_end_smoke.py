from __future__ import annotations

import json
from pathlib import Path

from scripts import benchmark_end_to_end_smoke


def test_all_public_benchmark_entrypoints_run_end_to_end(tmp_path: Path) -> None:
    report = benchmark_end_to_end_smoke.run(
        root=Path(__file__).resolve().parents[1],
        work_dir=tmp_path / "e2e",
        updates=2,
        batch_size=1,
        seed=13,
    )

    assert report["schema"] == "path-opd-three-benchmark-e2e-smoke-v1"
    assert report["status"] == "PASS"
    assert report["claim"] == "synthetic_three_benchmark_end_to_end_only"
    assert report["synthetic_only"] is True
    assert report["external_runtime_executed"] is False
    assert set(report["benchmarks"]) == set(benchmark_end_to_end_smoke.BENCHMARKS)
    for result in report["benchmarks"].values():
        assert result["train"]["exact_resume"] is True
        assert result["evaluate"]["all_contract_checks"] is True


def test_cli_writes_path_free_summary(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    assert (
        benchmark_end_to_end_smoke.main(
            [
                "--benchmark",
                "maniskill",
                "--output",
                str(output),
                "--work-dir",
                str(tmp_path / "work"),
                "--updates",
                "2",
                "--batch-size",
                "1",
                "--seed",
                "5",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["benchmarks"]["maniskill"]["evaluate"]["panel_rows"] == 320
    assert str(tmp_path) not in output.read_text(encoding="utf-8")

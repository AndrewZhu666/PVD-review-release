#!/usr/bin/env python3
"""Run all released benchmark train/evaluate entrypoints end to end.

This command is deliberately asset-free.  It launches the public
``--synthetic-smoke`` branches as subprocesses, then validates the reports
from each benchmark's train and evaluator boundary.  A passing report proves
the released Path-OPD/Endpoint-DAgger, checkpoint/resume, panel identity, and
action-trace contracts; it does not prove a real simulator or paper-scale
benchmark run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

BENCHMARKS = ("maniskill", "calvin_abc_d", "metaworld_mt50")
SCHEMA = "path-opd-three-benchmark-e2e-smoke-v1"


class EndToEndSmokeError(RuntimeError):
    """Raised when a public synthetic benchmark boundary fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EndToEndSmokeError(message)


def _read_report(path: Path, *, benchmark: str, operation: str) -> dict[str, Any]:
    if not path.is_file():
        raise EndToEndSmokeError(f"{benchmark} {operation} did not write {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EndToEndSmokeError(f"{benchmark} {operation} report is not valid JSON") from error
    _require(isinstance(payload, dict), f"{benchmark} {operation} report must be an object")
    return payload


def _run_entrypoint(
    root: Path,
    *,
    benchmark: str,
    operation: str,
    work_dir: Path,
    output: Path,
    updates: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    script = root / "benchmarks" / benchmark / f"{operation}.py"
    command = [
        sys.executable,
        str(script),
        "--synthetic-smoke",
        "--work-dir",
        str(work_dir),
        "--output",
        str(output),
        "--updates",
        str(updates),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
    ]
    environment = os.environ.copy()
    source = str(root / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise EndToEndSmokeError(
            f"{benchmark} {operation} synthetic entrypoint exited {completed.returncode}: {details}"
        )
    return _read_report(output, benchmark=benchmark, operation=operation)


def _validate_train(report: dict[str, Any], *, benchmark: str, updates: int) -> dict[str, Any]:
    _require(
        report.get("schema") == "path-opd-runner-synthetic-smoke-v1",
        f"{benchmark} train schema drifted",
    )
    _require(report.get("status") == "PASS", f"{benchmark} train did not pass")
    _require(
        report.get("claim") == "synthetic_runner_contract_smoke_only",
        f"{benchmark} train claim drifted",
    )
    _require(report.get("synthetic_only") is True, f"{benchmark} train is not synthetic-only")
    _require(
        report.get("runner_entrypoint_executed") is True,
        f"{benchmark} train entrypoint was not executed",
    )
    for key in (
        "external_runtime_executed",
        "external_assets_used",
        "external_openpi_executed",
        "real_simulator",
        "gpu_executed",
        "paper_scale",
    ):
        _require(report.get(key) is False, f"{benchmark} train evidence field {key} is not false")
    benchmark_report = report.get("benchmark_report")
    _require(isinstance(benchmark_report, dict), f"{benchmark} train benchmark report is missing")
    _require(
        benchmark_report.get("benchmark") == benchmark,
        f"{benchmark} train benchmark identity drifted",
    )
    _require(
        benchmark_report.get("config", {}).get("updates") == updates,
        f"{benchmark} train update budget drifted",
    )
    checks = benchmark_report.get("checks")
    _require(
        isinstance(checks, dict) and all(checks.values()),
        f"{benchmark} train contract checks failed",
    )
    _require(
        benchmark_report.get("interrupted", {}).get("exact_resume") is True,
        f"{benchmark} train resume proof failed",
    )
    _require(
        benchmark_report.get("uninterrupted", {}).get("next_update") == updates,
        f"{benchmark} train did not complete all updates",
    )
    return {
        "status": report["status"],
        "claim": report["claim"],
        "updates": updates,
        "all_contract_checks": True,
        "exact_resume": True,
    }


def _validate_evaluator(report: dict[str, Any], *, benchmark: str) -> dict[str, Any]:
    _require(
        report.get("schema") == "path-opd-evaluator-synthetic-smoke-v1",
        f"{benchmark} evaluator schema drifted",
    )
    _require(report.get("status") == "PASS", f"{benchmark} evaluator did not pass")
    _require(
        report.get("claim") == "synthetic_evaluator_contract_smoke_only",
        f"{benchmark} evaluator claim drifted",
    )
    _require(report.get("synthetic_only") is True, f"{benchmark} evaluator is not synthetic-only")
    _require(
        report.get("evaluator_entrypoint_executed") is True,
        f"{benchmark} evaluator entrypoint was not executed",
    )
    _require(
        report.get("training_smoke_executed") is True,
        f"{benchmark} evaluator skipped training/checkpoint proof",
    )
    for key in (
        "external_runtime_executed",
        "external_assets_used",
        "external_openpi_executed",
        "real_simulator",
        "gpu_executed",
        "paper_scale",
    ):
        _require(
            report.get(key) is False, f"{benchmark} evaluator evidence field {key} is not false"
        )
    checkpoint = report.get("checkpoint")
    panel = report.get("panel")
    checks = report.get("checks")
    _require(isinstance(checkpoint, dict), f"{benchmark} evaluator checkpoint proof is missing")
    _require(
        checkpoint.get("schema") == "path-opd-benchmark-synthetic-checkpoint-v1",
        f"{benchmark} evaluator checkpoint schema drifted",
    )
    _require(
        isinstance(panel, dict) and panel.get("formal") is False,
        f"{benchmark} evaluator panel must be explicitly non-formal",
    )
    action_trace = report.get("action_trace")
    _require(
        isinstance(action_trace, dict),
        f"{benchmark} evaluator action trace proof is missing",
    )
    _require(
        action_trace.get("panel_rows") == panel.get("rows"),
        f"{benchmark} evaluator did not consume every panel row",
    )
    _require(
        action_trace.get("ordered_identity_sha256") == panel.get("ordered_identity_sha256"),
        f"{benchmark} evaluator panel identity binding drifted",
    )
    _require(
        panel.get("rows") == panel.get("unique_identities"),
        f"{benchmark} evaluator panel identity denominator failed",
    )
    _require(
        isinstance(checks, dict) and all(checks.values()),
        f"{benchmark} evaluator contract checks failed",
    )
    training_report = report.get("training_report")
    _require(
        isinstance(training_report, dict)
        and training_report.get("checks", {}).get("exact_resume") is True,
        f"{benchmark} evaluator checkpoint reload proof failed",
    )
    return {
        "status": report["status"],
        "claim": report["claim"],
        "panel_rows": panel["rows"],
        "checkpoint_schema": checkpoint["schema"],
        "all_contract_checks": True,
    }


def run(
    *,
    root: Path,
    work_dir: Path,
    updates: int,
    batch_size: int,
    seed: int,
    benchmarks: tuple[str, ...] = BENCHMARKS,
) -> dict[str, Any]:
    if updates < 2 or batch_size < 1:
        raise EndToEndSmokeError("updates must be at least 2 and batch-size must be positive")
    if not benchmarks or any(benchmark not in BENCHMARKS for benchmark in benchmarks):
        raise EndToEndSmokeError(f"unsupported benchmark selection: {benchmarks!r}")
    root = root.resolve()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for benchmark in benchmarks:
        benchmark_dir = work_dir / benchmark
        train_report = _run_entrypoint(
            root,
            benchmark=benchmark,
            operation="train",
            work_dir=benchmark_dir / "train",
            output=benchmark_dir / "train.json",
            updates=updates,
            batch_size=batch_size,
            seed=seed,
        )
        evaluator_report = _run_entrypoint(
            root,
            benchmark=benchmark,
            operation="evaluate",
            work_dir=benchmark_dir / "evaluate",
            output=benchmark_dir / "evaluate.json",
            updates=updates,
            batch_size=batch_size,
            seed=seed,
        )
        results[benchmark] = {
            "train": _validate_train(train_report, benchmark=benchmark, updates=updates),
            "evaluate": _validate_evaluator(evaluator_report, benchmark=benchmark),
        }
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "claim": "synthetic_three_benchmark_end_to_end_only",
        "synthetic_only": True,
        "external_runtime_executed": False,
        "external_assets_used": False,
        "real_simulator": False,
        "gpu_executed": False,
        "paper_scale": False,
        "config": {"updates": updates, "batch_size": batch_size, "seed": seed},
        "benchmarks": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, action="append", dest="benchmarks")
    parser.add_argument(
        "--work-dir", type=Path, default=Path("artifacts/three-benchmark-e2e-smoke")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(
            root=Path(__file__).resolve().parents[1],
            work_dir=args.work_dir,
            updates=args.updates,
            batch_size=args.batch_size,
            seed=args.seed,
            benchmarks=tuple(args.benchmarks) if args.benchmarks else BENCHMARKS,
        )
    except (EndToEndSmokeError, OSError, subprocess.SubprocessError) as error:
        print(f"benchmark end-to-end smoke: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

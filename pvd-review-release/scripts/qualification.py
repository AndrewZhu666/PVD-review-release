#!/usr/bin/env python3
"""Run release qualification checks without requiring benchmark dependencies.

The script is deliberately conservative: a source-level or CPU check can pass
without implying that a simulator, CUDA runtime, container engine, or external
model asset is available.  Results are split into PASS, WARN, and BLOCKED so a
reviewer can see exactly which evidence is still missing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STATUSES = ("PASS", "WARN", "BLOCKED", "FAIL", "SKIP")
BENCHMARKS = ("maniskill", "calvin_abc_d", "metaworld_mt50")
EXPECTED_SYNTHETIC_PANEL_ROWS = {
    "maniskill": 320,
    "calvin_abc_d": 1000,
    "metaworld_mt50": 500,
}
REQUIRED_FILES = (
    "pyproject.toml",
    "src/path_opd/core.py",
    "src/path_opd/cli.py",
    "src/path_opd/adapters/benchmark_smoke.py",
    "configs/maniskill.json",
    "configs/calvin_abc_d.json",
    "configs/metaworld_mt50.json",
    "docker/Dockerfile",
    "docker/Dockerfile.dockerignore",
    "docker/entrypoint.sh",
    "scripts/audit_anonymity.py",
    "scripts/benchmark_environment_smoke.py",
    "scripts/benchmark_end_to_end_smoke.py",
    "scripts/qualification.py",
)
CONTAINER_ENGINES = ("docker", "podman", "buildah", "apptainer")
BENCHMARK_MODULES = {
    "maniskill": ("torch", "numpy", "gymnasium", "mani_skill"),
    "metaworld_mt50": ("torch", "numpy", "gymnasium", "mujoco", "metaworld"),
    "calvin_abc_d": ("torch", "numpy", "gymnasium"),
}
REDACT_PATH = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/)[^\s\"']+")


@dataclass(frozen=True)
class Check:
    """One independently reviewable qualification result."""

    check_id: str
    status: str
    evidence: str
    command: str | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unsupported check status: {self.status}")

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _python(root: Path) -> Path:
    candidate = root / ".venv" / "bin" / "python"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    return Path(sys.executable)


def _safe_text(value: str) -> str:
    """Keep command diagnostics from copying machine-specific paths into reports."""

    return REDACT_PATH.sub("<path>", value).replace("\n", " ").strip()[:240]


def _run(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> tuple[int | None, str, str]:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return None, "", "command not found"
    except subprocess.TimeoutExpired:
        return None, "", "timed out"
    return completed.returncode, _safe_text(completed.stdout), _safe_text(completed.stderr)


def _runtime_env(temp_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = str(temp_root / "pycache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _required_files(root: Path) -> Check:
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        return Check(
            "source_layout",
            "FAIL",
            "missing required files: " + ", ".join(missing),
        )
    return Check("source_layout", "PASS", f"all {len(REQUIRED_FILES)} required release files exist")


def _python_version(python: Path, root: Path) -> Check:
    code, stdout, stderr = _run([str(python), "--version"], cwd=root, timeout=15)
    if code != 0:
        detail = stderr or stdout or "interpreter failed"
        return Check("python_runtime", "BLOCKED", _safe_text(detail), "python --version")
    match = re.search(r"(\d+)\.(\d+)", stdout or stderr)
    if match is None:
        return Check("python_runtime", "WARN", "interpreter responded but version was not parsed")
    version = (int(match.group(1)), int(match.group(2)))
    status = "PASS" if version >= (3, 10) else "FAIL"
    return Check("python_runtime", status, f"Python {version[0]}.{version[1]} (requires >=3.10)")


def _run_python_check(
    check_id: str,
    python: Path,
    root: Path,
    arguments: list[str],
    *,
    command: str,
    timeout: float = 180.0,
) -> Check:
    with tempfile.TemporaryDirectory(prefix="path-opd-qualification-") as temporary:
        env = _runtime_env(Path(temporary))
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(root / "src"), env.get("PYTHONPATH", "")) if item
        )
        code, stdout, stderr = _run([str(python), *arguments], cwd=root, env=env, timeout=timeout)
    if code == 0:
        return Check(check_id, "PASS", "command completed successfully", command)
    detail = stderr or stdout or f"exit code {code}"
    return Check(check_id, "FAIL", _safe_text(detail), command)


def _cpu_checks(root: Path, python: Path, *, execute: bool) -> list[Check]:
    if not execute:
        return [
            Check(
                "pytest",
                "SKIP",
                "runtime checks disabled",
                "python -m pytest -p no:cacheprovider -q",
            ),
            Check("ruff", "SKIP", "runtime checks disabled", "ruff check --no-cache ."),
            Check(
                "compileall",
                "SKIP",
                "runtime checks disabled",
                "python -m compileall -q src benchmarks tests",
            ),
            Check(
                "cpu_smoke", "SKIP", "runtime checks disabled", "python -m path_opd.cli smoke ..."
            ),
            Check(
                "benchmark_synthetic_smoke",
                "SKIP",
                "runtime checks disabled",
                "python -m path_opd.cli benchmark smoke --benchmark all ...",
            ),
            Check(
                "benchmark_environment_smoke_contract",
                "SKIP",
                "runtime checks disabled",
                "python scripts/benchmark_environment_smoke.py --benchmark all ...",
            ),
            Check(
                "benchmark_end_to_end_smoke",
                "SKIP",
                "runtime checks disabled",
                "python scripts/benchmark_end_to_end_smoke.py ...",
            ),
            *[
                Check(
                    f"{operation}_synthetic_smoke_{benchmark}",
                    "SKIP",
                    "runtime checks disabled",
                    (
                        f"python benchmarks/{benchmark}/"
                        f"{'train.py' if operation == 'runner' else 'evaluate.py'} "
                        "--synthetic-smoke ..."
                    ),
                )
                for operation in ("runner", "evaluator")
                for benchmark in BENCHMARKS
            ],
            Check(
                "anonymity_scan",
                "SKIP",
                "runtime checks disabled",
                "python scripts/audit_anonymity.py . --no-git",
            ),
        ]

    checks = [
        _run_python_check(
            "pytest",
            python,
            root,
            ["-m", "pytest", "-p", "no:cacheprovider", "-q"],
            command="python -m pytest -p no:cacheprovider -q",
            timeout=300,
        ),
        _run_python_check(
            "compileall",
            python,
            root,
            ["-m", "compileall", "-q", "src", "benchmarks", "tests"],
            command="python -m compileall -q src benchmarks tests",
        ),
    ]
    ruff = root / ".venv" / "bin" / "ruff"
    if ruff.is_file() and os.access(ruff, os.X_OK):
        code, stdout, stderr = _run([str(ruff), "check", "--no-cache", "."], cwd=root, timeout=120)
        checks.append(
            Check(
                "ruff",
                "PASS" if code == 0 else "FAIL",
                "command completed successfully" if code == 0 else _safe_text(stderr or stdout),
                "ruff check --no-cache .",
            )
        )
    else:
        checks.append(
            _run_python_check(
                "ruff",
                python,
                root,
                ["-m", "ruff", "check", "--no-cache", "."],
                command="python -m ruff check --no-cache .",
            )
        )

    with tempfile.TemporaryDirectory(prefix="path-opd-smoke-") as temporary:
        work_dir = Path(temporary) / "smoke"
        output = work_dir / "report.json"
        env = _runtime_env(Path(temporary))
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(root / "src"), env.get("PYTHONPATH", "")) if item
        )
        code, stdout, stderr = _run(
            [
                str(python),
                "-m",
                "path_opd.cli",
                "smoke",
                "--work-dir",
                str(work_dir),
                "--output",
                str(output),
            ],
            cwd=root,
            env=env,
            timeout=180,
        )
        smoke_status = "FAIL"
        smoke_evidence = _safe_text(stderr or stdout or f"exit code {code}")
        if code == 0 and output.is_file():
            try:
                payload = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if (
                payload.get("status") == "PASS"
                and payload.get("interrupted", {}).get("exact_resume") is True
            ):
                smoke_status = "PASS"
                smoke_evidence = "deterministic toy smoke and exact interrupted resume passed"
        checks.append(
            Check("cpu_smoke", smoke_status, smoke_evidence, "python -m path_opd.cli smoke ...")
        )

        benchmark_work = Path(temporary) / "benchmark-smoke"
        benchmark_output = benchmark_work / "report.json"
        code, stdout, stderr = _run(
            [
                str(python),
                "-m",
                "path_opd.cli",
                "benchmark",
                "smoke",
                "--benchmark",
                "all",
                "--updates",
                "2",
                "--batch-size",
                "1",
                "--work-dir",
                str(benchmark_work),
                "--output",
                str(benchmark_output),
            ],
            cwd=root,
            env=env,
            timeout=180,
        )
        benchmark_status = "FAIL"
        benchmark_evidence = _safe_text(stderr or stdout or f"exit code {code}")
        if code == 0 and benchmark_output.is_file():
            try:
                benchmark_payload = json.loads(benchmark_output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                benchmark_payload = {}
            reports = benchmark_payload.get("benchmarks", {})
            if (
                benchmark_payload.get("status") == "PASS"
                and benchmark_payload.get("claim") == "synthetic_contract_smoke_only"
                and benchmark_payload.get("synthetic_only") is True
                and benchmark_payload.get("path_opd_core_executed") is True
                and benchmark_payload.get("checkpoint_reload_executed") is True
                and benchmark_payload.get("external_assets_used") is False
                and benchmark_payload.get("openpi_adapter_executed") is True
                and benchmark_payload.get("synthetic_openpi_adapter_executed") is True
                and benchmark_payload.get("external_openpi_executed") is False
                and benchmark_payload.get("endpoint_dagger_executed") is True
                and benchmark_payload.get("synthetic_endpoint_dagger_executed") is True
                and benchmark_payload.get("external_endpoint_dagger_executed") is False
                and benchmark_payload.get("real_simulator") is False
                and benchmark_payload.get("gpu_executed") is False
                and benchmark_payload.get("paper_scale") is False
                and set(reports) == set(BENCHMARKS)
                and all(report.get("status") == "PASS" for report in reports.values())
            ):
                benchmark_status = "PASS"
                benchmark_evidence = (
                    "all three configured benchmark contracts passed synthetic CPU smoke; "
                    "report is explicitly not simulator or paper-scale evidence"
                )
        checks.append(
            Check(
                "benchmark_synthetic_smoke",
                benchmark_status,
                benchmark_evidence,
                "python -m path_opd.cli benchmark smoke --benchmark all ...",
            )
        )

        # The environment probe must not guess machine-local paths.  With no
        # external arguments it should still run every synthetic algorithm
        # contract, then report each real simulator probe as BLOCKED.
        environment_work = Path(temporary) / "environment-smoke"
        environment_output = environment_work / "report.json"
        code, stdout, stderr = _run(
            [
                str(python),
                "scripts/benchmark_environment_smoke.py",
                "--benchmark",
                "all",
                "--output",
                str(environment_output),
            ],
            cwd=root,
            env=env,
            timeout=180,
        )
        environment_status = "FAIL"
        environment_evidence = _safe_text(stderr or stdout or f"exit code {code}")
        if code == 0 and environment_output.is_file():
            try:
                environment_payload = json.loads(environment_output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                environment_payload = {}
            environment_results = environment_payload.get("results", [])
            algorithm_reports = [
                item.get("checks", {}).get("algorithm_synthetic_contract", {})
                for item in environment_results
                if isinstance(item, dict)
            ]
            if (
                environment_payload.get("schema")
                == "path-opd-benchmark-environment-smoke-v1"
                and environment_payload.get("status") == "BLOCKED"
                and environment_payload.get("claim")
                == "synthetic_algorithm_plus_environment_one_step_smoke"
                and environment_payload.get("algorithm_synthetic_smoke_executed") is True
                and environment_payload.get("algorithm_synthetic_smoke_completed") is True
                and len(environment_results) == len(BENCHMARKS)
                and all(
                    item.get("status") == "BLOCKED" and item.get("blockers")
                    for item in environment_results
                )
                and all(
                    report.get("schema") == "path-opd-benchmark-synthetic-smoke-v2"
                    and report.get("status") == "PASS"
                    and report.get("synthetic_only") is True
                    and report.get("executed") is True
                    and report.get("completed") is True
                    and report.get("exact_resume") is True
                    and report.get("external_runtime_executed") is False
                    and report.get("external_assets_used") is False
                    and report.get("real_simulator") is False
                    and report.get("gpu_executed") is False
                    and report.get("paper_scale") is False
                    for report in algorithm_reports
                )
            ):
                environment_status = "PASS"
                environment_evidence = (
                    "all synthetic algorithm contracts passed and missing external paths "
                    "were reported as BLOCKED without path guessing"
                )
        checks.append(
            Check(
                "benchmark_environment_smoke_contract",
                environment_status,
                environment_evidence,
                "python scripts/benchmark_environment_smoke.py --benchmark all ...",
            )
        )

        # Launch all six public train/evaluate entrypoints through one aggregate
        # subprocess contract.  The aggregate validates every nested report and
        # gives qualification a single end-to-end result rather than six
        # unrelated green checks.
        end_to_end_work = Path(temporary) / "three-benchmark-e2e"
        end_to_end_output = end_to_end_work / "report.json"
        code, stdout, stderr = _run(
            [
                str(python),
                "scripts/benchmark_end_to_end_smoke.py",
                "--work-dir",
                str(end_to_end_work),
                "--output",
                str(end_to_end_output),
                "--updates",
                "2",
                "--batch-size",
                "1",
                "--seed",
                "13",
            ],
            cwd=root,
            env=env,
            timeout=300,
        )
        end_to_end_status = "FAIL"
        end_to_end_evidence = _safe_text(stderr or stdout or f"exit code {code}")
        end_to_end_payload: dict[str, Any] = {}
        if code == 0 and end_to_end_output.is_file():
            try:
                end_to_end_payload = json.loads(end_to_end_output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                end_to_end_payload = {}
        end_to_end_results = end_to_end_payload.get("benchmarks", {})
        top_level_contract = (
            end_to_end_payload.get("schema") == "path-opd-three-benchmark-e2e-smoke-v1"
            and end_to_end_payload.get("status") == "PASS"
            and end_to_end_payload.get("claim") == "synthetic_three_benchmark_end_to_end_only"
            and end_to_end_payload.get("synthetic_only") is True
            and end_to_end_payload.get("external_runtime_executed") is False
            and end_to_end_payload.get("external_assets_used") is False
            and end_to_end_payload.get("real_simulator") is False
            and end_to_end_payload.get("gpu_executed") is False
            and end_to_end_payload.get("paper_scale") is False
            and end_to_end_payload.get("config")
            == {"updates": 2, "batch_size": 1, "seed": 13}
            and isinstance(end_to_end_results, dict)
            and set(end_to_end_results) == set(BENCHMARKS)
        )
        per_benchmark_status: dict[str, tuple[bool, bool]] = {}
        if top_level_contract:
            for benchmark in BENCHMARKS:
                result = end_to_end_results.get(benchmark, {})
                train = result.get("train", {}) if isinstance(result, dict) else {}
                evaluator = result.get("evaluate", {}) if isinstance(result, dict) else {}
                train_passed = (
                    train.get("status") == "PASS"
                    and train.get("claim") == "synthetic_runner_contract_smoke_only"
                    and train.get("updates") == 2
                    and train.get("all_contract_checks") is True
                    and train.get("exact_resume") is True
                )
                evaluator_passed = (
                    evaluator.get("status") == "PASS"
                    and evaluator.get("claim") == "synthetic_evaluator_contract_smoke_only"
                    and evaluator.get("panel_rows")
                    == EXPECTED_SYNTHETIC_PANEL_ROWS[benchmark]
                    and evaluator.get("checkpoint_schema")
                    == "path-opd-benchmark-synthetic-checkpoint-v1"
                    and evaluator.get("all_contract_checks") is True
                )
                per_benchmark_status[benchmark] = (train_passed, evaluator_passed)
            if all(all(statuses) for statuses in per_benchmark_status.values()):
                end_to_end_status = "PASS"
                end_to_end_evidence = (
                    "all six public train/evaluate entrypoints passed one aggregate "
                    "synthetic subprocess contract; external runtime, assets, simulator, "
                    "GPU, and paper-scale execution are false"
                )
        checks.append(
            Check(
                "benchmark_end_to_end_smoke",
                end_to_end_status,
                end_to_end_evidence,
                "python scripts/benchmark_end_to_end_smoke.py ...",
            )
        )
        for benchmark in BENCHMARKS:
            train_passed, evaluator_passed = per_benchmark_status.get(
                benchmark, (False, False)
            )
            checks.extend(
                (
                    Check(
                        f"runner_synthetic_smoke_{benchmark}",
                        "PASS" if train_passed else "FAIL",
                        (
                            "public runner passed inside the aggregate synthetic "
                            "end-to-end contract"
                            if train_passed
                            else "aggregate end-to-end runner proof failed"
                        ),
                        f"python benchmarks/{benchmark}/train.py --synthetic-smoke ...",
                    ),
                    Check(
                        f"evaluator_synthetic_smoke_{benchmark}",
                        "PASS" if evaluator_passed else "FAIL",
                        (
                            "public evaluator consumed every synthetic panel row and "
                            "passed checkpoint/action-trace binding inside the aggregate contract"
                            if evaluator_passed
                            else "aggregate end-to-end evaluator proof failed"
                        ),
                        f"python benchmarks/{benchmark}/evaluate.py --synthetic-smoke ...",
                    ),
                )
            )

    checks.append(
        _run_python_check(
            "anonymity_scan",
            python,
            root,
            ["scripts/audit_anonymity.py", ".", "--no-git", "--format", "json"],
            command="python scripts/audit_anonymity.py . --no-git --format json",
        )
    )
    return checks


def _docker_checks(root: Path) -> list[Check]:
    dockerfile = root / "docker" / "Dockerfile"
    dockerignore = root / "docker" / "Dockerfile.dockerignore"
    checks: list[Check] = []
    if not dockerfile.is_file() or not dockerignore.is_file():
        return [Check("docker_static", "FAIL", "Dockerfile or Dockerfile.dockerignore is missing")]
    docker_text = dockerfile.read_text(encoding="utf-8")
    ignore_lines = {
        line.strip()
        for line in dockerignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    required_allowlist = {
        "!pyproject.toml",
        "!src/",
        "!configs/",
        "!tests/",
        "!docker/",
        "!scripts/",
        "!scripts/benchmark_end_to_end_smoke.py",
        "!scripts/benchmark_environment_smoke.py",
        "!scripts/qualification.py",
        "**/.venv/",
        "**/.pytest_cache/",
        "**/.ruff_cache/",
    }
    missing = sorted(required_allowlist - ignore_lines)
    if missing:
        checks.append(
            Check("docker_context_allowlist", "FAIL", "missing patterns: " + ", ".join(missing))
        )
    else:
        checks.append(
            Check(
                "docker_context_allowlist",
                "PASS",
                "context is deny-by-default and excludes local environments/caches",
            )
        )
    from_match = re.search(
        r"(?m)^\s*FROM(?:\s+--platform=\S+)?\s+(\S+)",
        docker_text,
    )
    base_ref = from_match.group(1) if from_match else ""
    if re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", base_ref):
        checks.append(Check("docker_base_digest", "PASS", "base image is digest pinned"))
    else:
        checks.append(
            Check(
                "docker_base_digest",
                "WARN",
                "base image uses a mutable tag; record a digest before archival",
            )
        )
    pinned_args = all(
        re.search(pattern, docker_text, flags=re.MULTILINE) is not None
        for pattern in (
            r"^ARG TORCH_VERSION=\S+$",
            r"^ARG PYTEST_VERSION=\S+$",
            r"^ARG SETUPTOOLS_VERSION=\S+$",
            r"^ARG WHEEL_VERSION=\S+$",
        )
    )
    checks.append(
        Check(
            "docker_tool_versions",
            "PASS" if pinned_args else "WARN",
            "tool version build arguments are exact"
            if pinned_args
            else "one or more tool versions are not exact",
        )
    )
    if "download.pytorch.org/whl/cpu" in docker_text:
        checks.append(
            Check(
                "docker_core_mode",
                "PASS",
                "image is intentionally CPU-only for portable core verification",
            )
        )
    else:
        checks.append(Check("docker_core_mode", "WARN", "CPU wheel source was not detected"))

    engine = next((name for name in CONTAINER_ENGINES if shutil.which(name)), None)
    if engine is None:
        checks.append(
            Check(
                "container_engine",
                "BLOCKED",
                "no Docker-compatible engine is installed; image build is unverified",
            )
        )
    else:
        code, stdout, stderr = _run([engine, "--version"], cwd=root, timeout=20)
        if code == 0:
            checks.append(
                Check(
                    "container_engine",
                    "PASS",
                    f"{engine} executable responds; run a clean build to close the gate",
                )
            )
        else:
            checks.append(
                Check(
                    "container_engine",
                    "BLOCKED",
                    f"{engine} is present but did not respond: {_safe_text(stderr or stdout)}",
                )
            )
    return checks


def _gpu_checks(root: Path, python: Path) -> list[Check]:
    checks: list[Check] = []
    nvidia = shutil.which("nvidia-smi")
    if nvidia is None:
        checks.append(Check("nvidia_driver", "BLOCKED", "nvidia-smi is not installed"))
    else:
        code, stdout, _stderr = _run([nvidia, "-L"], cwd=root, timeout=20)
        checks.append(
            Check(
                "nvidia_driver",
                "PASS" if code == 0 and stdout else "BLOCKED",
                "NVIDIA driver reports at least one device"
                if code == 0 and stdout
                else "nvidia-smi did not report a usable device",
                "nvidia-smi -L",
            )
        )
    code, stdout, _stderr = _run(
        [
            str(python),
            "-c",
            "import torch; print(int(torch.cuda.is_available())); print(torch.version.cuda or '')",
        ],
        cwd=root,
        timeout=30,
    )
    available = code == 0 and stdout.splitlines() and stdout.splitlines()[0].strip() == "1"
    if code != 0:
        checks.append(
            Check(
                "torch_cuda",
                "BLOCKED",
                "selected Python cannot import torch",
                "python -c 'import torch'",
            )
        )
    elif available:
        checks.append(Check("torch_cuda", "PASS", "torch reports CUDA available"))
    else:
        checks.append(Check("torch_cuda", "BLOCKED", "torch reports CUDA unavailable"))
    return checks


def _module_check(root: Path, python: Path, benchmark: str, module: str) -> Check:
    code, _stdout, _stderr = _run(
        [
            str(python),
            "-c",
            (
                "import importlib.util, sys; "
                "sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
            ),
            module,
        ],
        cwd=root,
        timeout=20,
    )
    return Check(
        f"dependency_{benchmark}_{module.replace('.', '_')}",
        "PASS" if code == 0 else "BLOCKED",
        f"{module} is importable"
        if code == 0
        else f"{module} is not available in the selected environment",
    )


def _asset_checks(root: Path) -> list[Check]:
    checks: list[Check] = []
    for benchmark in BENCHMARKS:
        path = root / "configs" / "assets" / f"{benchmark}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            assets = payload["assets"]
            required = [asset for asset in assets if asset.get("required") is True]
            known_hashes = [
                asset
                for asset in required
                if asset.get("integrity", {}).get("status") == "known"
                and isinstance(asset.get("integrity", {}).get("sha256"), str)
            ]
            public_routes = [
                asset
                for asset in required
                if asset.get("location", {}).get("public_url_status") == "known"
                and asset.get("location", {}).get("public_url")
            ]
            licenses = [
                asset for asset in required if asset.get("license", {}).get("status") == "known"
            ]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            checks.append(
                Check(f"assets_{benchmark}", "FAIL", f"manifest could not be parsed: {error}")
            )
            continue
        status = (
            "PASS"
            if len(known_hashes) == len(required)
            and len(public_routes) == len(required)
            and len(licenses) == len(required)
            else "BLOCKED"
        )
        checks.append(
            Check(
                f"assets_{benchmark}",
                status,
                (
                    f"{len(known_hashes)}/{len(required)} required hashes, "
                    f"{len(public_routes)}/{len(required)} public routes, "
                    f"{len(licenses)}/{len(required)} licenses recorded"
                ),
            )
        )
    return checks


def _runner_checks(root: Path, *, execute_runtime: bool) -> list[Check]:
    checks: list[Check] = []
    for benchmark in ("maniskill", "metaworld_mt50"):
        directory = root / "benchmarks" / benchmark
        expected = (directory / "train.py", directory / "evaluate.py")
        checks.append(
            Check(
                f"runner_{benchmark}",
                "PASS" if all(path.is_file() for path in expected) else "FAIL",
                "train and evaluate entry points are present"
                if all(path.is_file() for path in expected)
                else "runner entry point is missing",
            )
        )
    calvin_directory = root / "benchmarks" / "calvin_abc_d"
    calvin_expected = tuple(
        calvin_directory / name
        for name in ("common.py", "environment.py", "train.py", "evaluate.py")
    )
    calvin_present = all(path.is_file() for path in calvin_expected)
    checks.append(
        Check(
            "runner_calvin_abc_d",
            "BLOCKED" if calvin_present else "FAIL",
            (
                "CALVIN entry points are present, but paper-scale runtime qualification "
                "requires external assets, simulator, and GPU execution"
                if calvin_present
                else "CALVIN runner entry point is missing"
            ),
        )
    )
    calvin_status = (
        "BLOCKED" if execute_runtime and calvin_present else "SKIP" if calvin_present else "FAIL"
    )
    calvin_evidence = (
        "no successful eight-rank simulator/GPU/asset qualification artifact is "
        "available in this release environment"
        if execute_runtime and calvin_present
        else "real CALVIN qualification is not executed by static inspection"
        if calvin_present
        else "cannot qualify a missing CALVIN runner"
    )
    checks.append(
        Check(
            "qualification_calvin_abc_d",
            calvin_status,
            calvin_evidence,
            (
                "torchrun --standalone --nproc-per-node=8 "
                "benchmarks/calvin_abc_d/train.py --max-updates 1 ..."
            ),
        )
    )
    return checks


def _release_hygiene_checks(root: Path) -> list[Check]:
    generated_directories: list[str] = []
    bytecode_files = 0
    generated_directory_names = {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "artifacts",
        "build",
        "dist",
    }
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_names[:] = [name for name in directory_names if name != ".git"]
        base = Path(directory)
        for name in directory_names:
            if name in generated_directory_names or name.endswith(".egg-info"):
                generated_directories.append((base / name).relative_to(root).as_posix())
        bytecode_files += sum(
            1 for name in filenames if Path(name).suffix.lower() in {".pyc", ".pyo"}
        )
    generated_directories.sort()
    generated_summary: list[str] = generated_directories[:12]
    if len(generated_directories) > len(generated_summary):
        generated_summary.append(f"(+{len(generated_directories) - 12} more directories)")
    if bytecode_files:
        generated_summary.append(f"{bytecode_files} bytecode files")
    generated = ", ".join(generated_summary)
    checks = [
        Check(
            "generated_artifacts",
            "WARN" if generated_summary else "PASS",
            "remove local generated state before archiving: " + generated
            if generated_summary
            else "no common generated directories found",
        ),
        Check(
            "git_history",
            "PASS" if (root / ".git").exists() else "WARN",
            "repository history is available for audit"
            if (root / ".git").exists()
            else (
                "no Git history in the release directory; create a fresh "
                "anonymous history before submission"
            ),
        ),
    ]
    return checks


def qualify(root: Path, *, execute_runtime: bool = True) -> dict[str, Any]:
    root = root.resolve()
    python = _python(root)
    checks: list[Check] = [_required_files(root), _python_version(python, root)]
    checks.extend(_docker_checks(root))
    checks.extend(_gpu_checks(root, python))
    checks.extend(_runner_checks(root, execute_runtime=execute_runtime))
    checks.extend(_asset_checks(root))
    checks.extend(_release_hygiene_checks(root))
    checks.extend(_cpu_checks(root, python, execute=execute_runtime))
    for benchmark, modules in BENCHMARK_MODULES.items():
        for module in modules:
            checks.append(_module_check(root, python, benchmark, module))

    counts = {
        status.lower(): sum(check.status == status for check in checks) for status in STATUSES
    }
    blocking = counts["blocked"] + counts["fail"]
    overall = "QUALIFIED" if blocking == 0 and counts["warn"] == 0 else "NOT_QUALIFIED"
    return {
        "schema": "path-opd-qualification-v1",
        "python": platform.python_version(),
        "runtime_checks_executed": execute_runtime,
        "checks": [check.as_dict() for check in checks],
        "summary": {
            **counts,
            "overall": overall,
            "blocking_checks": blocking,
        },
        "claim_boundary": {
            "core_cpu": "supported when CPU checks pass",
            "paper_scale": (
                "withheld until container, GPU, external assets, and all benchmark runners qualify"
            ),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Release Qualification Report",
        "",
        (
            "This report is generated by `scripts/qualification.py`. It separates "
            "source/CPU evidence from external runtime gates; `NOT_QUALIFIED` is "
            "expected until every blocking gate is closed."
        ),
        "",
        f"- Overall: **{payload['summary']['overall']}**",
        f"- Runtime checks executed: `{payload['runtime_checks_executed']}`",
        (
            f"- Counts: PASS `{payload['summary']['pass']}`, "
            f"WARN `{payload['summary']['warn']}`, "
            f"BLOCKED `{payload['summary']['blocked']}`, "
            f"FAIL `{payload['summary']['fail']}`, "
            f"SKIP `{payload['summary']['skip']}`"
        ),
        "",
        "| Check | Status | Evidence | Command |",
        "|---|---|---|---|",
    ]
    for check in payload["checks"]:
        evidence = str(check["evidence"]).replace("|", "\\|")
        command = str(check.get("command") or "").replace("|", "\\|")
        lines.append(
            f"| `{check['check_id']}` | **{check['status']}** | {evidence} | `{command}` |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- CPU core evidence does not imply simulator or paper-scale reproduction.",
            "- A clean source tree still needs a successful clean container build and run.",
            (
                "- Missing assets, licenses, runtime evidence, or completed "
                "CALVIN simulator/evaluator qualification remain explicit blockers."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-runtime", action="store_true", help="only inspect static contracts")
    parser.add_argument(
        "--strict", action="store_true", help="return non-zero when any gate is not closed"
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    payload = qualify(args.root, execute_runtime=not args.skip_runtime)
    rendered = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_markdown(payload)
    )
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 1 if args.strict and payload["summary"]["overall"] != "QUALIFIED" else 0


if __name__ == "__main__":
    raise SystemExit(main())

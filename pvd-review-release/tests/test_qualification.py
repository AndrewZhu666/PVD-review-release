from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "qualification.py"
SPEC = importlib.util.spec_from_file_location("path_opd_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualification
SPEC.loader.exec_module(qualification)


def test_static_qualification_has_explicit_claim_boundary() -> None:
    report = qualification.qualify(REPOSITORY_ROOT, execute_runtime=False)

    assert report["schema"] == "path-opd-qualification-v1"
    assert report["runtime_checks_executed"] is False
    assert report["claim_boundary"]["paper_scale"].startswith("withheld")
    checks = {item["check_id"]: item for item in report["checks"]}
    assert checks["docker_context_allowlist"]["status"] == "PASS"
    assert checks["runner_maniskill"]["status"] == "PASS"
    assert checks["runner_metaworld_mt50"]["status"] == "PASS"
    assert checks["runner_calvin_abc_d"]["status"] == "BLOCKED"
    assert "asset" in checks["runner_calvin_abc_d"]["evidence"]
    assert checks["qualification_calvin_abc_d"]["status"] == "SKIP"
    assert checks["benchmark_end_to_end_smoke"]["status"] == "SKIP"
    for benchmark in qualification.BENCHMARKS:
        assert checks[f"runner_synthetic_smoke_{benchmark}"]["status"] == "SKIP"
        assert checks[f"evaluator_synthetic_smoke_{benchmark}"]["status"] == "SKIP"
    dockerignore = (REPOSITORY_ROOT / "docker" / "Dockerfile.dockerignore").read_text(
        encoding="utf-8"
    )
    entrypoint = (REPOSITORY_ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    assert "!scripts/qualification.py" in dockerignore
    assert "!scripts/benchmark_environment_smoke.py" in dockerignore
    assert "!scripts/benchmark_end_to_end_smoke.py" in dockerignore
    assert "qualification)" in entrypoint


def test_docker_base_check_requires_digest_on_from_instruction(tmp_path: Path) -> None:
    docker = tmp_path / "docker"
    docker.mkdir()
    (docker / "Dockerfile").write_text(
        "# misleading @sha256:" + "0" * 64 + "\nFROM python:3.11-slim-bookworm\n",
        encoding="utf-8",
    )
    (docker / "Dockerfile.dockerignore").write_text(
        "\n".join(
            (
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
            )
        ),
        encoding="utf-8",
    )

    checks = {item.check_id: item for item in qualification._docker_checks(tmp_path)}

    assert checks["docker_base_digest"].status == "WARN"


def test_release_docker_base_uses_a_complete_digest() -> None:
    checks = {
        item.check_id: item for item in qualification._docker_checks(REPOSITORY_ROOT)
    }

    assert checks["docker_base_digest"].status == "PASS"


def test_markdown_renderer_keeps_gate_status_visible() -> None:
    payload = {
        "runtime_checks_executed": False,
        "summary": {
            "overall": "NOT_QUALIFIED",
            "pass": 1,
            "warn": 0,
            "blocked": 1,
            "fail": 0,
            "skip": 0,
        },
        "checks": [
            {
                "check_id": "container_engine",
                "status": "BLOCKED",
                "evidence": "image build is unverified",
                "command": "docker build ...",
            }
        ],
    }

    rendered = qualification.render_markdown(payload)

    assert "**NOT_QUALIFIED**" in rendered
    assert "`container_engine`" in rendered
    assert "**BLOCKED**" in rendered

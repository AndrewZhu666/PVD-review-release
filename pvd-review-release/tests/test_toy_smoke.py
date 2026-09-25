"""End-to-end CPU proof for optimization and exact checkpoint resume."""

from __future__ import annotations

from pathlib import Path

import torch

from path_opd.adapters.toy import ToyRunConfig, run_smoke


def test_cpu_smoke_optimizes_and_resumes_bit_exactly(tmp_path: Path) -> None:
    config = ToyRunConfig(updates=6, batch_size=2, seed=7)

    result = run_smoke(tmp_path, config)

    assert result["schema"] == "path-opd-toy-smoke-v1"
    assert result["status"] == "PASS"
    assert result["config"]["updates"] == 6
    assert result["uninterrupted"]["next_update"] == 6
    assert result["interrupted"]["checkpoint_update"] == 3
    assert result["interrupted"]["exact_resume"] is True
    assert (
        result["interrupted"]["resumed_final_sha256"] == result["uninterrupted"]["student_sha256"]
    )
    assert result["uninterrupted"]["final_loss"] < result["uninterrupted"]["initial_loss"]
    assert result["checks"] == {
        "loss_decreased": True,
        "teacher_unchanged": True,
        "endpoint_excluded": True,
        "path_states_detached": True,
    }

    checkpoint = torch.load(tmp_path / "toy_checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["schema"] == "path-opd-toy-checkpoint-v1"
    assert checkpoint["next_update"] == 3
    assert checkpoint["config"] == result["config"]
    assert checkpoint["teacher_sha256"] == result["uninterrupted"]["teacher_sha256"]


def test_cpu_smoke_is_deterministic_across_fresh_workspaces(tmp_path: Path) -> None:
    config = ToyRunConfig(updates=4, batch_size=2, seed=11)

    first = run_smoke(tmp_path / "first", config)
    second = run_smoke(tmp_path / "second", config)

    assert first["uninterrupted"] == second["uninterrupted"]
    assert first["interrupted"] == second["interrupted"]
    assert first["checks"] == second["checks"]


def test_parameter_hash_does_not_require_numpy() -> None:
    """The release runtime must hash tensors in a minimal torch environment."""
    from path_opd.integrity import parameter_sha256

    model = torch.nn.Linear(2, 1)
    digest = parameter_sha256(model)
    assert len(digest) == 64

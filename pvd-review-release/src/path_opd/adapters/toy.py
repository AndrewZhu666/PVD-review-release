"""Deterministic CPU reproduction of the Path-OPD update and resume contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from path_opd.core import ActionContract, FlowSchedule, FrozenTeacher, PathOPD
from path_opd.integrity import parameter_sha256


@dataclass(frozen=True)
class ToyRunConfig:
    steps: int = 4
    horizon: int = 3
    action_dims: int = 2
    model_action_dims: int = 4
    batch_size: int = 4
    updates: int = 40
    learning_rate: float = 0.03
    seed: int = 7

    def __post_init__(self) -> None:
        values = (
            self.steps,
            self.horizon,
            self.action_dims,
            self.model_action_dims,
            self.batch_size,
            self.updates,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Toy run dimensions and updates must be positive.")
        if self.action_dims > self.model_action_dims:
            raise ValueError("Physical action dimensions exceed model action dimensions.")


class ToyVelocityModel(nn.Module):
    def __init__(self, action_dims: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.empty(action_dims))
        self.time_scale = nn.Parameter(torch.empty(action_dims))
        self.bias = nn.Parameter(torch.empty(action_dims))

    def reset(self, generator: torch.Generator) -> None:
        with torch.no_grad():
            self.scale.copy_(torch.randn(self.scale.shape, generator=generator) * 0.2)
            self.time_scale.copy_(torch.randn(self.time_scale.shape, generator=generator) * 0.2)
            self.bias.copy_(torch.randn(self.bias.shape, generator=generator) * 0.2)

    def forward(self, states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        shape = (states.shape[0],) + (1,) * (states.ndim - 2) + (1,)
        time = times.reshape(shape)
        return states * self.scale + time * self.time_scale + self.bias


def _teacher(action_dims: int) -> ToyVelocityModel:
    model = ToyVelocityModel(action_dims)
    with torch.no_grad():
        model.scale.copy_(torch.linspace(0.25, 0.55, action_dims))
        model.time_scale.copy_(torch.linspace(-0.35, 0.15, action_dims))
        model.bias.copy_(torch.linspace(0.1, -0.2, action_dims))
    return model


def _student(config: ToyRunConfig) -> ToyVelocityModel:
    generator = torch.Generator().manual_seed(config.seed + 1)
    model = ToyVelocityModel(config.model_action_dims)
    model.reset(generator)
    return model


def _rollout(
    student: ToyVelocityModel,
    noise: torch.Tensor,
    schedule: FlowSchedule,
) -> torch.Tensor:
    state = noise
    chains = [state]
    delta = 1.0 / schedule.steps
    for time_value in schedule.times:
        times = torch.full((state.shape[0],), time_value, dtype=state.dtype)
        state = state - delta * student(state, times)
        chains.append(state)
    return torch.stack(chains, dim=1)


def _save_checkpoint(
    path: Path,
    *,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    next_update: int,
    config: ToyRunConfig,
    teacher_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "path-opd-toy-checkpoint-v1",
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(),
            "next_update": next_update,
            "config": asdict(config),
            "teacher_sha256": teacher_hash,
        },
        path,
    )


def _train(
    config: ToyRunConfig,
    *,
    stop_after: int | None = None,
    checkpoint: Path | None = None,
    resume: Path | None = None,
) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    student = _student(config)
    teacher_model = _teacher(config.model_action_dims)
    teacher = FrozenTeacher(teacher_model, lambda model, states, times: model(states, times))
    teacher_hash = parameter_sha256(teacher_model)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 2)
    next_update = 0
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload["schema"] != "path-opd-toy-checkpoint-v1":
            raise RuntimeError("Unsupported toy checkpoint schema.")
        if payload["config"] != asdict(config):
            raise RuntimeError("Toy checkpoint configuration mismatch.")
        if payload["teacher_sha256"] != teacher_hash:
            raise RuntimeError("Toy teacher hash mismatch.")
        student.load_state_dict(payload["student"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        generator.set_state(payload["generator"])
        next_update = int(payload["next_update"])
    schedule = FlowSchedule.uniform(config.steps)
    objective = PathOPD(
        ActionContract(config.horizon, config.action_dims),
        schedule,
    )
    initial_loss: float | None = None
    final_loss: float | None = None
    end_update = config.updates if stop_after is None else min(stop_after, config.updates)
    last_trace = None
    for update in range(next_update, end_update):
        noise = torch.randn(
            config.batch_size,
            config.horizon,
            config.model_action_dims,
            generator=generator,
        )
        chains = _rollout(student, noise, schedule)
        optimizer.zero_grad(set_to_none=True)
        result = objective.supervise(chains, student, teacher)
        result.loss.backward()
        optimizer.step()
        value = float(result.loss.detach())
        initial_loss = value if initial_loss is None else initial_loss
        final_loss = value
        last_trace = result.trace
        if checkpoint is not None and update + 1 == end_update:
            _save_checkpoint(
                checkpoint,
                student=student,
                optimizer=optimizer,
                generator=generator,
                next_update=update + 1,
                config=config,
                teacher_hash=teacher_hash,
            )
    teacher.assert_unchanged()
    return {
        "student": student,
        "student_sha256": parameter_sha256(student),
        "teacher_sha256": teacher_hash,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "next_update": end_update,
        "endpoint_excluded": bool(last_trace is not None and last_trace.steps == config.steps),
        "path_states_detached": bool(
            last_trace is not None and not last_trace.states.requires_grad
        ),
    }


def run_smoke(output_dir: Path, config: ToyRunConfig | None = None) -> dict[str, Any]:
    """Run uninterrupted and interrupted CPU training, then compare exact state."""
    config = config or ToyRunConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    split = max(1, config.updates // 2)
    checkpoint = output_dir / "toy_checkpoint.pt"
    uninterrupted = _train(config)
    first_half = _train(config, stop_after=split, checkpoint=checkpoint)
    resumed = _train(config, resume=checkpoint)
    exact_resume = uninterrupted["student_sha256"] == resumed["student_sha256"]
    if not exact_resume:
        raise RuntimeError("Interrupted toy run diverged from uninterrupted training.")
    if uninterrupted["teacher_sha256"] != resumed["teacher_sha256"]:
        raise RuntimeError("Toy teacher changed across interruption.")
    if uninterrupted["final_loss"] is None or uninterrupted["initial_loss"] is None:
        raise RuntimeError("Toy smoke produced no optimization steps.")
    checks = {
        "loss_decreased": uninterrupted["final_loss"] < uninterrupted["initial_loss"],
        "teacher_unchanged": uninterrupted["teacher_sha256"] == resumed["teacher_sha256"],
        "endpoint_excluded": uninterrupted["endpoint_excluded"],
        "path_states_detached": uninterrupted["path_states_detached"],
    }
    if not all(checks.values()) or not exact_resume:
        raise RuntimeError(f"Toy smoke invariant failed: {checks}")
    return {
        "schema": "path-opd-toy-smoke-v1",
        "status": "PASS",
        "config": asdict(config),
        "uninterrupted": {key: value for key, value in uninterrupted.items() if key != "student"},
        "interrupted": {
            "checkpoint_update": first_half["next_update"],
            "resumed_final_sha256": resumed["student_sha256"],
            "exact_resume": exact_resume,
        },
        "checks": checks,
    }

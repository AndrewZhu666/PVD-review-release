"""Behavioral tests for the Path-OPD supervision invariants."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import path_opd.core as path_core
from path_opd import (
    ACTION_EXPERT_PREFIXES,
    ActionContract,
    FlowSchedule,
    FrozenTeacher,
    PathOPD,
    RolloutTrace,
    configure_action_expert,
    validate_exact_trace,
)


class _ScalarVelocity(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value))

    def forward(self, states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        del times
        return states * self.scale


def test_rollout_trace_excludes_endpoint_and_detaches_student_path() -> None:
    chains = torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[999.0]]]], requires_grad=True)
    schedule = FlowSchedule.uniform(3)

    trace = RolloutTrace.from_chains(chains, schedule)

    torch.testing.assert_close(
        trace.states, torch.tensor([[[1.0]], [[2.0]], [[3.0]]]), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(trace.times, torch.tensor([1.0, 2.0 / 3.0, 1.0 / 3.0]))
    assert not trace.states.requires_grad
    assert trace.states.grad_fn is None
    assert 999.0 not in trace.states


def test_rollout_trace_uses_float32_flow_times_for_reduced_precision_chains() -> None:
    chains = torch.zeros(1, 3, 1, 1, dtype=torch.float16)
    trace = RolloutTrace.from_chains(chains, FlowSchedule.uniform(2))
    assert trace.times.dtype is torch.float32


def test_supervision_cannot_backpropagate_through_sampled_rollout() -> None:
    chains = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]], dtype=torch.float32, requires_grad=True)
    student_scale = nn.Parameter(torch.tensor(0.5))
    seen_student_states: list[torch.Tensor] = []
    seen_teacher_states: list[torch.Tensor] = []

    def student(states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        del times
        seen_student_states.append(states)
        return states * student_scale

    def teacher(states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        del times
        seen_teacher_states.append(states)
        return torch.zeros_like(states)

    objective = PathOPD(ActionContract(1, 1), FlowSchedule.uniform(2))
    result = objective.supervise(chains, student, teacher)
    result.loss.backward()

    assert student_scale.grad is not None
    assert float(student_scale.grad) == pytest.approx(2.5)
    assert chains.grad is None
    assert seen_student_states[0] is result.trace.states
    assert seen_teacher_states[0] is result.trace.states
    assert not seen_student_states[0].requires_grad
    torch.testing.assert_close(
        seen_student_states[0], torch.tensor([[[1.0]], [[2.0]]]), rtol=0.0, atol=0.0
    )


def test_frozen_teacher_is_eval_only_no_grad_and_immutable() -> None:
    model = _ScalarVelocity(2.0)
    model.train()
    query_grad_modes: list[bool] = []

    def query(
        teacher_model: nn.Module,
        states: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        query_grad_modes.append(torch.is_grad_enabled())
        return teacher_model(states, times)

    teacher = FrozenTeacher(model, query)
    states = torch.ones(2, 1, 1, requires_grad=True)
    velocity = teacher.velocity(states, torch.tensor([1.0, 0.5]))

    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert query_grad_modes == [False]
    assert not velocity.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    teacher.assert_unchanged()


def test_supervision_rejects_teacher_parameter_mutation() -> None:
    model = _ScalarVelocity(2.0)

    def mutating_query(
        teacher_model: nn.Module,
        states: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            teacher_model.scale.add_(1.0)  # type: ignore[attr-defined]
        return teacher_model(states, times)

    teacher = FrozenTeacher(model, mutating_query)
    objective = PathOPD(ActionContract(1, 1), FlowSchedule.uniform(2))
    chains = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])

    with pytest.raises(RuntimeError, match="Frozen teacher parameters changed"):
        objective.supervise(chains, lambda states, times: states * 0.0, teacher)


def test_frozen_teacher_rejects_data_and_buffer_mutation() -> None:
    model = _ScalarVelocity(2.0)
    model.register_buffer("marker", torch.tensor(1.0))
    teacher = FrozenTeacher(model, lambda module, states, times: module(states, times))

    with torch.no_grad():
        model.scale.data.add_(1.0)
    with pytest.raises(RuntimeError, match="Frozen teacher parameters changed"):
        teacher.assert_unchanged()

    model = _ScalarVelocity(2.0)
    model.register_buffer("marker", torch.tensor(1.0))
    teacher = FrozenTeacher(model, lambda module, states, times: module(states, times))
    with torch.no_grad():
        model.marker.add_(1.0)
    with pytest.raises(RuntimeError, match="Frozen teacher parameters changed"):
        teacher.assert_unchanged()


def test_fingerprint_handles_large_tensors_in_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Module()
    model.register_buffer("payload", torch.arange(4097, dtype=torch.float32))
    monkeypatch.setattr(path_core, "_FINGERPRINT_CHUNK_BYTES", 17)
    original_tolist = torch.Tensor.tolist
    converted_sizes: list[int] = []

    def bounded_tolist(tensor: torch.Tensor) -> list[int]:
        converted_sizes.append(tensor.numel())
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", bounded_tolist)

    first = path_core._module_fingerprint(model)
    with torch.no_grad():
        model.payload[2048] += 1
    second = path_core._module_fingerprint(model)

    assert first != second
    assert converted_sizes
    assert max(converted_sizes) <= 17
    assert len(converted_sizes) > 2


def test_fast_teacher_guard_can_skip_deep_fingerprint() -> None:
    model = _ScalarVelocity(2.0)
    teacher = FrozenTeacher(
        model,
        lambda module, states, times: module(states, times),
        deep_fingerprint=False,
    )
    teacher.assert_unchanged()
    with pytest.raises(RuntimeError, match="Deep teacher fingerprint"):
        teacher.assert_unchanged(deep=True)


def test_default_action_expert_selector_is_exact_and_freezes_other_families() -> None:
    model = nn.ModuleDict(
        {
            "action_in_proj": nn.Linear(1, 1),
            "backbone": nn.Linear(1, 1),
            "critic": nn.Linear(1, 1),
        }
    )
    selected = configure_action_expert(model)
    assert selected == ("action_in_proj.weight", "action_in_proj.bias")
    assert ACTION_EXPERT_PREFIXES
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("action_in_proj.")
    )


def test_exact_trace_rejects_non_finite_provenance() -> None:
    chains = torch.zeros(1, 3, 1, 1)
    with pytest.raises(FloatingPointError, match="model_actions"):
        validate_exact_trace(
            chains,
            torch.tensor([[[float("nan")]]]),
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
            expected_steps=2,
        )

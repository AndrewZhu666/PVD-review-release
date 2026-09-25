"""Scientific core of Path-OPD.

The module deliberately knows nothing about a simulator or VLA implementation.
Adapters supply velocity queries; this module owns path selection, detachment,
shape validation, action-domain cropping, and the distillation objective.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

VelocityQuery = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Canonical OpenPI action-expert families used by the archived experiments.
# Keeping the list here makes the trainable scope auditable across adapters.
ACTION_EXPERT_PREFIXES = (
    "paligemma_with_expert.gemma_expert.model.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
    "state_proj.",
    "action_time_mlp_in.",
    "action_time_mlp_out.",
)


# Keep explicit deep audits bounded in Python memory.  A teacher can contain
# billions of parameters; converting one full byte tensor with ``tolist``
# creates one Python integer object per byte and can exhaust host memory.
_FINGERPRINT_CHUNK_BYTES = 1024 * 1024


def _module_fingerprint(model: nn.Module) -> str:
    """Hash parameters and buffers without making NumPy a runtime dependency."""
    digest = hashlib.sha256()
    for kind, items in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in sorted(items):
            source_dtype = tensor.dtype
            value = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
            digest.update(kind.encode("ascii"))
            digest.update(name.encode("utf-8"))
            digest.update(str(source_dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            for start in range(0, value.numel(), _FINGERPRINT_CHUNK_BYTES):
                chunk = value[start : start + _FINGERPRINT_CHUNK_BYTES].cpu()
                digest.update(bytes(chunk.tolist()))
    return digest.hexdigest()


def _module_fast_state(model: nn.Module) -> tuple[tuple[Any, ...], ...]:
    """Capture metadata and version counters suitable for the training hot path."""
    entries: list[tuple[Any, ...]] = []
    for kind, items in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        entries.extend(
            (
                kind,
                name,
                id(tensor),
                str(tensor.dtype),
                tuple(tensor.shape),
                tensor._version,
            )
            for name, tensor in sorted(items)
        )
    entries.extend(
        ("module", name, id(module), module.training)
        for name, module in sorted(model.named_modules())
    )
    return tuple(entries)


@dataclass(frozen=True)
class ActionContract:
    """Executable action positions and physical dimensions."""

    chunk_length: int
    action_dims: int

    def __post_init__(self) -> None:
        if self.chunk_length <= 0 or self.action_dims <= 0:
            raise ValueError("Action dimensions must be positive.")

    def crop(self, tensor: torch.Tensor) -> torch.Tensor:
        """Remove non-executed positions and model-only action dimensions."""
        if tensor.ndim < 2:
            raise ValueError(f"Expected action tensor rank >= 2, got {tensor.ndim}.")
        if tensor.shape[-2] < self.chunk_length:
            raise ValueError(
                f"Action horizon {tensor.shape[-2]} is shorter than "
                f"chunk length {self.chunk_length}."
            )
        if tensor.shape[-1] < self.action_dims:
            raise ValueError(
                f"Action width {tensor.shape[-1]} is shorter than "
                f"physical width {self.action_dims}."
            )
        return tensor[..., : self.chunk_length, : self.action_dims]


@dataclass(frozen=True)
class FlowSchedule:
    """Pre-transition flow times and their normalized loss weights."""

    times: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.times:
            raise ValueError("A flow schedule must contain at least one transition.")
        if len(self.times) != len(self.weights):
            raise ValueError("One loss weight is required per flow transition.")
        if any(left <= right for left, right in zip(self.times, self.times[1:], strict=False)):
            raise ValueError("Flow times must be strictly descending.")
        values = torch.tensor((*self.times, *self.weights), dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError("Flow times and weights must be finite.")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("Flow weights must be non-negative.")
        if abs(sum(self.weights) - 1.0) > 1e-12:
            raise ValueError("Flow weights must sum to one.")

    @property
    def steps(self) -> int:
        return len(self.times)

    @classmethod
    def uniform(cls, steps: int) -> FlowSchedule:
        """Create the Euler schedule used in all reported experiments."""
        if not isinstance(steps, int) or steps < 1:
            raise ValueError(f"steps must be a positive integer, got {steps!r}.")
        return cls(
            times=tuple(1.0 - index / steps for index in range(steps)),
            weights=tuple(1.0 / steps for _ in range(steps)),
        )


@dataclass(frozen=True)
class RolloutTrace:
    """Detached student-owned pre-transition states."""

    states: torch.Tensor
    times: torch.Tensor
    batch_size: int
    steps: int

    @classmethod
    def from_chains(
        cls,
        chains: torch.Tensor,
        schedule: FlowSchedule,
    ) -> RolloutTrace:
        """Exclude the endpoint and detach the sampled student path."""
        if chains.ndim != 4:
            raise ValueError(
                f"Expected chains [batch, steps+1, horizon, action_dim], got {tuple(chains.shape)}."
            )
        batch_size, chain_states, horizon, action_dim = chains.shape
        if chain_states != schedule.steps + 1:
            raise ValueError(f"Expected {schedule.steps + 1} chain states, got {chain_states}.")
        if batch_size <= 0 or horizon <= 0 or action_dim <= 0:
            raise ValueError("Rollout chains must have positive dimensions.")
        if not torch.isfinite(chains).all():
            raise FloatingPointError("Rollout chains contain NaN or Inf.")
        states = chains[:, :-1].reshape(
            batch_size * schedule.steps,
            horizon,
            action_dim,
        )
        states = states.detach().clone()
        times = torch.tensor(
            schedule.times,
            device=chains.device,
            # Historical OpenPI workers query NFT with float32 timesteps even
            # when the sampled action chain is held in reduced precision.
            dtype=torch.float32,
        ).repeat(batch_size)
        return cls(states=states, times=times, batch_size=batch_size, steps=schedule.steps)

    def unflatten_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        """Restore a velocity query to [batch, flow_time, horizon, dim]."""
        if velocity.ndim != 3:
            raise ValueError(
                "A velocity query must return [batch*steps, horizon, action_dim], got "
                f"{tuple(velocity.shape)}."
            )
        expected = self.batch_size * self.steps
        if velocity.shape[0] != expected:
            raise ValueError(
                f"Velocity query returned {velocity.shape[0]} rows; expected {expected}."
            )
        return velocity.reshape(self.batch_size, self.steps, *velocity.shape[1:])


class FrozenTeacher:
    """Freeze a teacher and expose actor-only, no-gradient velocity queries."""

    def __init__(
        self,
        model: nn.Module,
        query: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        deep_fingerprint: bool = True,
    ) -> None:
        """Freeze ``model`` and optionally capture a byte-level audit baseline.

        Large benchmark teachers should set ``deep_fingerprint=False``.  Their
        training hot path still checks module identity, tensor metadata, and
        PyTorch version counters without copying every weight to host memory.
        """
        self.model = model.eval()
        self.model.requires_grad_(False)
        self._query = query
        self._initial_fast_state = _module_fast_state(self.model)
        self._deep_fingerprint_enabled = deep_fingerprint
        self._initial_fingerprint = (
            _module_fingerprint(self.model) if deep_fingerprint else None
        )

    def velocity(self, states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Query teacher velocity with autograd disabled."""
        with torch.no_grad():
            velocity = self._query(self.model, states, times)
        if not isinstance(velocity, torch.Tensor):
            raise TypeError("Teacher velocity query must return a tensor.")
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise RuntimeError("Frozen teacher acquired gradients.")
        if not torch.isfinite(velocity).all():
            raise FloatingPointError("Teacher velocity contains NaN or Inf.")
        return velocity.detach()

    def bind(
        self, query: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> FrozenTeacher:
        """Reuse this teacher identity with a new rollout-specific query.

        Benchmark workers create a new OpenPI context for every rollout.  Reusing
        the already-captured identity avoids hashing and copying the full teacher
        model on every update while retaining the fast-state guard in
        :meth:`PathOPD.supervise`.
        """
        bound = object.__new__(FrozenTeacher)
        bound.model = self.model
        bound._query = query
        bound._initial_fast_state = self._initial_fast_state
        bound._deep_fingerprint_enabled = self._deep_fingerprint_enabled
        bound._initial_fingerprint = self._initial_fingerprint
        return bound

    def assert_unchanged(self, *, deep: bool | None = None) -> None:
        """Check teacher identity; deep checks also hash parameter and buffer bytes."""
        if _module_fast_state(self.model) != self._initial_fast_state:
            raise RuntimeError("Frozen teacher parameters changed.")
        if deep is None:
            deep = self._deep_fingerprint_enabled
        if deep:
            if self._initial_fingerprint is None:
                raise RuntimeError("Deep teacher fingerprint was not captured.")
            if _module_fingerprint(self.model) != self._initial_fingerprint:
                raise RuntimeError("Frozen teacher parameters changed.")


@dataclass(frozen=True)
class PathLoss:
    """Loss and audit evidence returned by one Path-OPD supervision call."""

    loss: torch.Tensor
    per_time: torch.Tensor
    trace: RolloutTrace
    student_velocity: torch.Tensor
    teacher_velocity: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.loss
        yield self.per_time


class PathOPD:
    """Path-OPD objective behind one benchmark-independent interface."""

    def __init__(self, contract: ActionContract, schedule: FlowSchedule) -> None:
        self.contract = contract
        self.schedule = schedule

    def loss_from_velocities(
        self,
        student_velocity: torch.Tensor,
        teacher_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time-weighted MSE on the executable physical action domain."""
        if student_velocity.shape != teacher_velocity.shape:
            raise ValueError(
                "Student/teacher velocity shape mismatch: "
                f"{tuple(student_velocity.shape)} != {tuple(teacher_velocity.shape)}."
            )
        if student_velocity.ndim != 4:
            raise ValueError(
                "Expected [batch, flow_time, horizon, action_dim], got "
                f"{tuple(student_velocity.shape)}."
            )
        if student_velocity.shape[1] != self.schedule.steps:
            raise ValueError(
                f"Expected {self.schedule.steps} flow transitions, got {student_velocity.shape[1]}."
            )
        student_valid = self.contract.crop(student_velocity)
        teacher_valid = self.contract.crop(teacher_velocity).detach()
        if not torch.isfinite(student_valid).all():
            raise FloatingPointError("Student velocity contains NaN or Inf.")
        if not torch.isfinite(teacher_valid).all():
            raise FloatingPointError("Teacher velocity contains NaN or Inf.")
        per_time = (student_valid - teacher_valid).square().mean(dim=(0, 2, 3))
        weights = torch.tensor(
            self.schedule.weights,
            device=per_time.device,
            dtype=per_time.dtype,
        )
        loss = torch.sum(per_time * weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("Path-OPD loss is NaN or Inf.")
        return loss, per_time

    def supervise(
        self,
        chains: torch.Tensor,
        student: VelocityQuery,
        teacher: VelocityQuery | FrozenTeacher,
    ) -> PathLoss:
        """Query both models on detached student states and return Path-OPD loss."""
        trace = RolloutTrace.from_chains(chains, self.schedule)
        teacher_query = teacher.velocity if isinstance(teacher, FrozenTeacher) else teacher
        with torch.no_grad():
            teacher_flat = teacher_query(trace.states, trace.times)
        student_flat = student(trace.states, trace.times)
        if not isinstance(student_flat, torch.Tensor) or not isinstance(teacher_flat, torch.Tensor):
            raise TypeError("Velocity queries must return tensors.")
        student_velocity = trace.unflatten_velocity(student_flat)
        teacher_velocity = trace.unflatten_velocity(teacher_flat).detach()
        loss, per_time = self.loss_from_velocities(student_velocity, teacher_velocity)
        if isinstance(teacher, FrozenTeacher):
            teacher.assert_unchanged(deep=False)
        return PathLoss(
            loss=loss,
            per_time=per_time,
            trace=trace,
            student_velocity=student_velocity,
            teacher_velocity=teacher_velocity,
        )


def configure_action_expert(
    model: nn.Module,
    prefixes: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Freeze a model and select only named action-expert parameter families."""
    prefixes = ACTION_EXPERT_PREFIXES if prefixes is None else tuple(prefixes)
    if not prefixes:
        raise ValueError("At least one action-expert prefix is required.")
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(tuple(prefixes)))
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise RuntimeError("No action-expert parameters were selected.")
    forbidden = [
        name
        for name in trainable
        if any(token in name.lower() for token in ("value", "critic", "noise"))
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden trainable parameters: {forbidden}")
    return tuple(trainable)


def validate_exact_trace(
    chains: torch.Tensor,
    model_actions: torch.Tensor,
    executed_actions: torch.Tensor,
    recorded_actions: torch.Tensor,
    *,
    expected_steps: int,
    tolerance: float = 0.0,
) -> dict[str, float | bool]:
    """Check solver endpoint and environment-action provenance."""
    if expected_steps < 1:
        raise ValueError("expected_steps must be positive.")
    if chains.ndim != 4:
        raise ValueError(
            f"Expected chains [batch, steps+1, horizon, action_dim], got {tuple(chains.shape)}."
        )
    if chains.shape[1] != expected_steps + 1:
        raise ValueError(f"Expected {expected_steps + 1} chain states, got {chains.shape[1]}.")
    if model_actions.shape != chains[:, -1].shape:
        raise ValueError("Model actions must match the chain endpoint shape.")
    if executed_actions.numel() != recorded_actions.numel():
        raise ValueError("Executed and recorded actions must have equal element counts.")
    try:
        executed = executed_actions.reshape_as(recorded_actions)
    except RuntimeError as error:
        raise ValueError("Executed and recorded action shapes are incompatible.") from error
    if tolerance < 0 or not torch.isfinite(torch.tensor(tolerance)):
        raise ValueError("tolerance must be finite and non-negative.")
    for name, tensor in (
        ("chains", chains),
        ("model_actions", model_actions),
        ("executed_actions", executed),
        ("recorded_actions", recorded_actions),
    ):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains NaN or Inf.")
    endpoint_delta = (chains[:, -1] - model_actions).abs().max().item()
    action_delta = (executed - recorded_actions).abs().max().item()
    if endpoint_delta > tolerance or action_delta > tolerance:
        raise RuntimeError(
            "Exact trace provenance failed: "
            f"endpoint_delta={endpoint_delta}, action_delta={action_delta}, "
            f"tolerance={tolerance}."
        )
    return {
        "endpoint_max_abs": endpoint_delta,
        "executed_action_max_abs": action_delta,
        "bitwise_endpoint": bool(torch.equal(chains[:, -1], model_actions)),
        "bitwise_executed_action": bool(torch.equal(executed, recorded_actions)),
    }


def require_tensor(value: Any, name: str) -> torch.Tensor:
    """Return a tensor or raise an adapter-facing error."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor.")
    return value

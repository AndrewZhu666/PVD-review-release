"""Dependency-light synthetic smoke for the three benchmark contracts.

The paper workers need external simulators, model checkpoints, and RLinf. This
module intentionally replaces those pieces with a deterministic tensor policy
and a tiny in-process environment. It still exercises the released
``PathOPD`` objective, the repository's ``OpenPIAdapter`` boundary, the native
Endpoint-DAgger call, each benchmark's action/flow dimensions, optimizer state,
checkpoint resume, and endpoint-to-environment action contract. It does not
execute an external RLinf/OpenPI checkout, a simulator, or any external asset.
Its reports are synthetic contract evidence, never benchmark qualification or
benchmark scores.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from path_opd.adapters.openpi import OpenPIAdapter
from path_opd.config import PAPER_BENCHMARKS, BenchmarkConfig, load_benchmark_config
from path_opd.core import (
    ActionContract,
    FlowSchedule,
    FrozenTeacher,
    PathOPD,
    validate_exact_trace,
)

SCHEMA = "path-opd-benchmark-synthetic-smoke-v2"
RUNNER_SCHEMA = "path-opd-runner-synthetic-smoke-v1"
EVALUATOR_SCHEMA = "path-opd-evaluator-synthetic-smoke-v1"


def _evidence_boundary() -> dict[str, bool | str]:
    return {
        "claim": "synthetic_contract_smoke_only",
        "synthetic_only": True,
        "path_opd_core_executed": True,
        "checkpoint_reload_executed": True,
        "external_assets_used": False,
        "openpi_adapter_executed": True,
        "synthetic_openpi_adapter_executed": True,
        "external_openpi_executed": False,
        "endpoint_dagger_executed": True,
        "synthetic_endpoint_dagger_executed": True,
        "external_endpoint_dagger_executed": False,
        "real_simulator": False,
        "gpu_executed": False,
        "paper_scale": False,
    }


@dataclass(frozen=True)
class SyntheticSmokeConfig:
    """Small deterministic budget used by the public smoke command."""

    updates: int = 4
    batch_size: int = 2
    seed: int = 7

    def __post_init__(self) -> None:
        if isinstance(self.updates, bool) or self.updates < 2:
            raise ValueError("synthetic smoke updates must be at least two")
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("synthetic smoke batch_size must be positive")


class _SyntheticOpenPIModel(nn.Module):
    """A tiny OpenPI-compatible action expert with no external assets.

    The wrapper implements the two calls used by the released workers: NFT
    velocity queries and native SFT action-chunk loss.  It is intentionally
    small, but the smoke therefore exercises the same adapter boundary as a
    real RLinf/OpenPI model.
    """

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.empty(dimensions))
        self.time_scale = nn.Parameter(torch.empty(dimensions))
        self.bias = nn.Parameter(torch.empty(dimensions))
        self.last_dagger_target: torch.Tensor | None = None

    def reset(self, generator: torch.Generator) -> None:
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.15)

    def _velocity(self, states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        time = times.reshape((states.shape[0],) + (1,) * (states.ndim - 1))
        return states * self.scale + time * self.time_scale + self.bias

    def rollout_velocity(self, states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Evaluate the synthetic flow model while sampling a rollout."""
        return self._velocity(states, times)

    def prepare_dagger_sft_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        target = batch["model_action"]
        if not isinstance(target, torch.Tensor):
            raise TypeError("synthetic DAgger target must be a tensor")
        self.last_dagger_target = target
        return {"target": target}

    def forward(self, **kwargs: Any) -> Any:
        nft_inputs = kwargs.get("nft_inputs")
        if isinstance(nft_inputs, Mapping):
            states = nft_inputs.get("x_t")
            times = nft_inputs.get("timesteps")
            if not isinstance(states, torch.Tensor) or not isinstance(times, torch.Tensor):
                raise TypeError("synthetic NFT inputs must contain tensor x_t and timesteps")
            return {"v_theta": self._velocity(states, times)}
        data = kwargs.get("data")
        if isinstance(data, Mapping):
            target = data.get("target")
            if not isinstance(target, torch.Tensor) or target.ndim != 2:
                raise TypeError("synthetic SFT target must be a flattened tensor")
            batch_size = target.shape[0]
            horizon = target.shape[1] // self.bias.shape[0]
            if horizon < 1 or horizon * self.bias.shape[0] != target.shape[1]:
                raise ValueError("synthetic SFT target has incompatible action dimensions")
            zeros = torch.zeros(
                batch_size,
                horizon,
                self.bias.shape[0],
                dtype=target.dtype,
                device=target.device,
            )
            predicted = self._velocity(zeros, torch.zeros(batch_size, device=target.device))
            return (predicted.reshape_as(target) - target).square().mean()
        raise ValueError("synthetic OpenPI model received an unsupported forward call")


class _SyntheticEnvironment:
    """Minimal environment boundary that consumes physical action prefixes."""

    def __init__(self, dimensions: int) -> None:
        self.state = torch.zeros(dimensions)
        self.steps = 0
        self._received_actions: list[torch.Tensor] = []

    def reset(self) -> torch.Tensor:
        self.state.zero_()
        self.steps = 0
        self._received_actions.clear()
        return self.state.clone()

    def step(self, action: torch.Tensor) -> None:
        if action.ndim != 1 or action.shape[0] != self.state.shape[0]:
            raise ValueError("synthetic environment received an invalid physical action")
        # Clone at the boundary so provenance reflects what the environment
        # actually received, rather than a caller-owned view that may change.
        received = action.detach().cpu().clone()
        self.state.add_(received)
        self._received_actions.append(received)
        self.steps += 1

    def recorded_actions(self) -> torch.Tensor:
        """Return the physical actions received since the last reset."""
        if not self._received_actions:
            return torch.empty((0, self.state.shape[0]), dtype=self.state.dtype)
        return torch.stack(self._received_actions).clone()


def _tensor_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(bytes(value.reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()


def _state_sha256(value: Any) -> str:
    """Hash nested tensor state without relying on pickle or NumPy."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(bytes(tensor.reshape(-1).view(torch.uint8).tolist()))
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda entry: (type(entry).__qualname__, repr(entry))):
                update(key)
                update(item[key])
            return
        if isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
            digest.update(type(item).__qualname__.encode("ascii"))
            digest.update(b"\0")
            for entry in item:
                update(entry)
            return
        if item is None or isinstance(item, bool | int | float | str):
            digest.update(type(item).__qualname__.encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(item).encode("utf-8"))
            return
        raise TypeError(f"unsupported synthetic checkpoint state: {type(item).__qualname__}")

    update(value)
    return digest.hexdigest()


def _teacher(dimensions: int) -> _SyntheticOpenPIModel:
    model = _SyntheticOpenPIModel(dimensions)
    with torch.no_grad():
        model.scale.copy_(torch.linspace(0.2, 0.5, dimensions))
        model.time_scale.copy_(torch.linspace(-0.3, 0.2, dimensions))
        model.bias.copy_(torch.linspace(0.1, -0.15, dimensions))
    return model


def _student(config: BenchmarkConfig, seed: int) -> _SyntheticOpenPIModel:
    generator = torch.Generator().manual_seed(seed + 1)
    model = _SyntheticOpenPIModel(config.action.model_dimensions)
    model.reset(generator)
    return model


def _rollout(
    student: _SyntheticOpenPIModel,
    noise: torch.Tensor,
    schedule: FlowSchedule,
) -> torch.Tensor:
    state = noise
    chains = [state]
    delta = 1.0 / schedule.steps
    for time_value in schedule.times:
        times = torch.full((state.shape[0],), time_value, dtype=state.dtype)
        state = state - delta * student.rollout_velocity(state, times)
        chains.append(state)
    return torch.stack(chains, dim=1)


def _probe_loss(
    student: _SyntheticOpenPIModel,
    teacher: FrozenTeacher,
    objective: PathOPD,
    noise: torch.Tensor,
    teacher_context: dict[str, Any],
) -> float:
    with torch.no_grad():
        chains = _rollout(student, noise, objective.schedule)
        rollout = _rollout_result(student, chains)
        teacher_context.clear()
        teacher_context.update(rollout["forward_inputs"])
        adapter = OpenPIAdapter.from_rollout(
            student,
            rollout,
            steps=objective.schedule.steps,
            nft_forward_type="NFT",
            sft_forward_type="SFT",
            contract=objective.contract,
        )
        return float(objective.supervise(adapter.chains, adapter.velocity, teacher).loss)


def _probe_endpoint_dagger(
    config: BenchmarkConfig,
    smoke_config: SyntheticSmokeConfig,
) -> dict[str, bool | float]:
    """Exercise the native SFT baseline branch on one synthetic rollout."""

    student = _student(config, smoke_config.seed + 10)
    teacher = _teacher(config.action.model_dimensions)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    schedule = FlowSchedule.uniform(config.flow.student_steps)
    contract = ActionContract(config.action.execution_prefix, config.action.physical_dimensions)
    generator = torch.Generator().manual_seed(smoke_config.seed + 11)
    student_noise = torch.randn(
        smoke_config.batch_size,
        config.action.model_horizon,
        config.action.model_dimensions,
        generator=generator,
    )
    teacher_noise = torch.randn(
        smoke_config.batch_size,
        config.action.model_horizon,
        config.action.model_dimensions,
        generator=generator,
    )
    student_chains = _rollout(student, student_noise, schedule)
    teacher_chains = _rollout(teacher, teacher_noise, schedule)
    rollout = _rollout_result(student, student_chains)
    adapter = OpenPIAdapter.from_rollout(
        student,
        rollout,
        steps=schedule.steps,
        nft_forward_type="NFT",
        sft_forward_type="SFT",
        contract=contract,
    )
    teacher_actions = teacher_chains[:, -1].detach().requires_grad_(True)
    loss = adapter.endpoint_dagger_loss(teacher_actions)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("synthetic Endpoint-DAgger loss is not a finite scalar")
    loss.backward()
    student_gradients = [parameter.grad for parameter in student.parameters()]
    student_gradient = any(
        gradient is not None and bool(torch.isfinite(gradient).all()) and bool(gradient.abs().sum())
        for gradient in student_gradients
    )
    target = student.last_dagger_target
    target_detached = target is not None and not target.requires_grad
    if teacher_actions.grad is not None:
        raise RuntimeError("synthetic Endpoint-DAgger target received a gradient")
    if not student_gradient or not target_detached:
        raise RuntimeError(
            "synthetic Endpoint-DAgger contract failed: "
            f"student_gradient={student_gradient}, target_detached={target_detached}"
        )
    return {
        "loss_finite": True,
        "target_detached": target_detached,
        "student_gradient": student_gradient,
        "loss": float(loss.detach()),
    }


def _rollout_result(
    model: _SyntheticOpenPIModel,
    chains: torch.Tensor,
) -> dict[str, Any]:
    """Package a synthetic rollout in the host model's OpenPI schema."""
    batch_size = chains.shape[0]
    model_actions = chains[:, -1].detach()
    return {
        "forward_inputs": {
            "chains": chains,
            "observation/state": torch.zeros(batch_size, 1, dtype=chains.dtype),
            "tokenized_prompt": torch.zeros(batch_size, 1, dtype=torch.long),
            "tokenized_prompt_mask": torch.ones(batch_size, 1, dtype=torch.bool),
            "action": model_actions,
        },
        "model_actions": model_actions,
    }


def _save_checkpoint(
    path: Path,
    *,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    next_update: int,
    benchmark: str,
    smoke_config: SyntheticSmokeConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "path-opd-benchmark-synthetic-checkpoint-v1",
            "benchmark": benchmark,
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(),
            "next_update": next_update,
            "smoke_config": asdict(smoke_config),
        },
        path,
    )


def _train(
    benchmark: str,
    config: BenchmarkConfig,
    smoke_config: SyntheticSmokeConfig,
    *,
    stop_after: int | None = None,
    resume: Path | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    student = _student(config, smoke_config.seed)
    teacher_model = _teacher(config.action.model_dimensions)
    schedule = FlowSchedule.uniform(config.flow.student_steps)
    objective = PathOPD(
        ActionContract(config.action.execution_prefix, config.action.physical_dimensions),
        schedule,
    )
    teacher_context: dict[str, Any] = {}

    def teacher_velocity(
        model: nn.Module,
        states: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        if not teacher_context:
            raise RuntimeError("synthetic teacher context was not bound")
        adapter = OpenPIAdapter(
            model=model,
            forward_inputs=teacher_context,
            steps=schedule.steps,
            nft_forward_type="NFT",
            contract=objective.contract,
        )
        return adapter.velocity(states, times)

    teacher = FrozenTeacher(teacher_model, teacher_velocity)
    teacher_hash = _tensor_sha256(teacher_model)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.training.optimizer.learning_rate,
        betas=config.training.optimizer.betas,
        eps=config.training.optimizer.epsilon,
        weight_decay=config.training.optimizer.weight_decay,
    )
    generator = torch.Generator().manual_seed(smoke_config.seed + 2)
    next_update = 0
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema") != "path-opd-benchmark-synthetic-checkpoint-v1":
            raise RuntimeError("synthetic benchmark checkpoint schema mismatch")
        if (
            payload.get("benchmark") != benchmark
            or payload.get("smoke_config") != asdict(smoke_config)
        ):
            raise RuntimeError("synthetic benchmark checkpoint contract mismatch")
        student.load_state_dict(payload["student"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        generator.set_state(payload["generator"])
        next_update = int(payload["next_update"])

    environment = _SyntheticEnvironment(config.action.physical_dimensions)
    environment.reset()
    probe_generator = torch.Generator().manual_seed(smoke_config.seed + 3)
    probe_noise = torch.randn(
        smoke_config.batch_size,
        config.action.model_horizon,
        config.action.model_dimensions,
        generator=probe_generator,
    )
    initial_loss = _probe_loss(student, teacher, objective, probe_noise, teacher_context)
    endpoint_trace: dict[str, float | bool] | None = None
    end_update = (
        smoke_config.updates
        if stop_after is None
        else min(stop_after, smoke_config.updates)
    )
    for update in range(next_update, end_update):
        noise = torch.randn(
            smoke_config.batch_size,
            config.action.model_horizon,
            config.action.model_dimensions,
            generator=generator,
        )
        chains = _rollout(student, noise, schedule)
        rollout = _rollout_result(student, chains)
        teacher_context.clear()
        teacher_context.update(rollout["forward_inputs"])
        student_adapter = OpenPIAdapter.from_rollout(
            student,
            rollout,
            steps=schedule.steps,
            nft_forward_type="NFT",
            sft_forward_type="SFT",
            contract=objective.contract,
        )
        optimizer.zero_grad(set_to_none=True)
        result = objective.supervise(student_adapter.chains, student_adapter.velocity, teacher)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            student.parameters(),
            config.training.optimizer.global_gradient_clip_norm,
        )
        optimizer.step()
        if update + 1 == end_update:
            model_actions = chains[:, -1].detach()
            executed = objective.contract.crop(model_actions)
            # This synthetic environment represents one rollout, so submit
            # the first batch element and validate against its recorded input.
            environment_actions = executed[0]
            for action in executed[0]:
                environment.step(action)
            recorded = environment.recorded_actions()
            endpoint_trace = validate_exact_trace(
                chains.detach(),
                model_actions,
                environment_actions,
                recorded,
                expected_steps=config.flow.student_steps,
            )
        if checkpoint is not None and update + 1 == end_update:
            _save_checkpoint(
                checkpoint,
                student=student,
                optimizer=optimizer,
                generator=generator,
                next_update=update + 1,
                benchmark=benchmark,
                smoke_config=smoke_config,
            )
    teacher.assert_unchanged()
    if endpoint_trace is None:
        raise RuntimeError("synthetic benchmark smoke performed no update")
    final_loss = _probe_loss(student, teacher, objective, probe_noise, teacher_context)
    return {
        "student_sha256": _tensor_sha256(student),
        "optimizer_sha256": _state_sha256(optimizer.state_dict()),
        "generator_sha256": _state_sha256(generator.get_state()),
        "teacher_sha256": teacher_hash,
        "teacher_unchanged": True,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "next_update": end_update,
        "endpoint_trace": endpoint_trace,
        "environment": {
            "steps": environment.steps,
            "received_action_count": int(environment.recorded_actions().shape[0]),
            "expected_action_sha256": _state_sha256(environment_actions),
            "received_action_sha256": _state_sha256(environment.recorded_actions()),
            "state": environment.state.tolist(),
        },
    }


def run_benchmark_smoke(
    benchmark: str,
    output_dir: Path,
    config: SyntheticSmokeConfig | None = None,
) -> dict[str, Any]:
    """Run one benchmark's configured Path-OPD contract on synthetic data."""

    if benchmark not in PAPER_BENCHMARKS:
        raise ValueError(f"unsupported benchmark {benchmark!r}")
    config = config or SyntheticSmokeConfig()
    benchmark_config = load_benchmark_config(benchmark)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = max(1, config.updates // 2)
    checkpoint = output_dir / f"{benchmark}-checkpoint.pt"
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        uninterrupted = _train(benchmark, benchmark_config, config)
        first_half = _train(
            benchmark,
            benchmark_config,
            config,
            stop_after=split,
            checkpoint=checkpoint,
        )
        resumed = _train(benchmark, benchmark_config, config, resume=checkpoint)
        endpoint_dagger = _probe_endpoint_dagger(benchmark_config, config)
    finally:
        torch.use_deterministic_algorithms(old_deterministic)
    exact_model_resume = uninterrupted["student_sha256"] == resumed["student_sha256"]
    exact_optimizer_resume = uninterrupted["optimizer_sha256"] == resumed["optimizer_sha256"]
    exact_generator_resume = uninterrupted["generator_sha256"] == resumed["generator_sha256"]
    exact_resume = exact_model_resume and exact_optimizer_resume and exact_generator_resume
    checks = {
        "loss_decreased": uninterrupted["final_loss"] < uninterrupted["initial_loss"],
        "teacher_unchanged": all(
            run["teacher_unchanged"]
            and run["teacher_sha256"] == uninterrupted["teacher_sha256"]
            for run in (uninterrupted, first_half, resumed)
        ),
        "endpoint_contract": bool(uninterrupted["endpoint_trace"]["bitwise_endpoint"]),
        "physical_action_contract": bool(
            uninterrupted["endpoint_trace"]["bitwise_executed_action"]
        ),
        "exact_model_resume": exact_model_resume,
        "exact_optimizer_resume": exact_optimizer_resume,
        "exact_generator_resume": exact_generator_resume,
        "exact_resume": exact_resume,
        "endpoint_dagger_loss_finite": bool(endpoint_dagger["loss_finite"]),
        "endpoint_dagger_target_detached": bool(endpoint_dagger["target_detached"]),
        "endpoint_dagger_student_gradient": bool(endpoint_dagger["student_gradient"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"synthetic benchmark smoke failed: {checks}")
    return {
        "schema": SCHEMA,
        "status": "PASS",
        **_evidence_boundary(),
        "benchmark": benchmark,
        "contract": {
            "flow_steps": benchmark_config.flow.student_steps,
            "model_horizon": benchmark_config.action.model_horizon,
            "execution_prefix": benchmark_config.action.execution_prefix,
            "physical_dimensions": benchmark_config.action.physical_dimensions,
        },
        "config": asdict(config),
        "uninterrupted": {
            key: value for key, value in uninterrupted.items() if key != "endpoint_trace"
        },
        "interrupted": {
            "checkpoint_update": first_half["next_update"],
            "resumed_final_sha256": resumed["student_sha256"],
            "exact_model_resume": exact_model_resume,
            "exact_optimizer_resume": exact_optimizer_resume,
            "exact_generator_resume": exact_generator_resume,
            "exact_resume": exact_resume,
        },
        "endpoint_dagger": endpoint_dagger,
        "checks": checks,
    }


def run_runner_synthetic_smoke(
    benchmark: str,
    output_dir: Path,
    *,
    entrypoint: str,
    config: SyntheticSmokeConfig | None = None,
) -> dict[str, Any]:
    """Exercise one released benchmark entrypoint's synthetic contract.

    The benchmark workers keep their real runtime deliberately lazy because
    model, simulator, and RLinf imports are optional.  Their ``--synthetic-
    smoke`` branch calls this helper, so the same deterministic tensor policy
    and environment proof is reachable through the public runner command.
    The returned evidence is intentionally separate from a real run: no
    external checkout, asset, simulator, CUDA device, or paper budget is used.
    """

    report = run_benchmark_smoke(benchmark, output_dir, config=config)
    return {
        "schema": RUNNER_SCHEMA,
        "status": "PASS",
        "claim": "synthetic_runner_contract_smoke_only",
        "synthetic_only": True,
        "runner_entrypoint_executed": True,
        "entrypoint": entrypoint,
        "external_runtime_executed": False,
        "external_assets_used": False,
        "external_openpi_executed": False,
        "real_simulator": False,
        "gpu_executed": False,
        "paper_scale": False,
        "benchmark_report": report,
    }


def _synthetic_panel(
    benchmark: str,
    config: BenchmarkConfig,
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Create and validate a deterministic panel at the evaluator boundary.

    The panel is deliberately not a benchmark fixture.  It only proves that
    an evaluator can bind an ordered denominator and stable row identities to
    a digest without importing a simulator or reading external assets.
    """

    rows = [
        {
            "row_index": index,
            "identity_sha256": hashlib.sha256(
                f"{benchmark}:synthetic:{seed}:{index}".encode()
            ).hexdigest(),
        }
        for index in range(config.evaluation.rows_per_model)
    ]
    identities = [row["identity_sha256"] for row in rows]
    if len(identities) != config.evaluation.rows_per_model:
        raise RuntimeError("synthetic evaluator panel denominator drifted")
    if len(set(identities)) != len(identities):
        raise RuntimeError("synthetic evaluator panel contains duplicate identities")
    payload = {
        "schema": "path-opd-synthetic-evaluator-panel-v1",
        "benchmark": benchmark,
        "seed": seed,
        "rows": rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    ordered_identity_digest = hashlib.sha256(
        json.dumps(identities, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    panel_path = output_dir / "synthetic-panel.json"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.write_bytes(encoded + b"\n")
    return {
        "schema": payload["schema"],
        "formal": False,
        "rows": len(rows),
        "unique_identities": len(set(identities)),
        "sha256": digest,
        "ordered_identity_sha256": ordered_identity_digest,
        "artifact": panel_path.name,
        # Keep the parsed rows in-process so the evaluator smoke can prove
        # that every panel identity is actually consumed by an action trace.
        # This private field is removed before the report is serialized.
        "_rows": rows,
    }


def _synthetic_action_trace(
    benchmark: str,
    config: BenchmarkConfig,
    *,
    seed: int,
    panel_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Exercise endpoint/action provenance for every evaluator panel row.

    A single synthetic rollout is enough to test tensor shapes, but it would
    not prove that an evaluator consumes its panel denominator.  Generate a
    batch with one trace per ordered panel row, then record each row through a
    fresh environment episode.  The resulting identity digest is reported so
    the caller can bind the traces to the panel it just validated.
    """

    generator = torch.Generator().manual_seed(seed + 97)
    row_count = len(panel_rows)
    if row_count < 1:
        raise RuntimeError("synthetic evaluator panel has no rows")
    identities = [row.get("identity_sha256") for row in panel_rows]
    if any(not isinstance(identity, str) for identity in identities):
        raise RuntimeError("synthetic evaluator panel rows lack identity_sha256")
    if len(set(identities)) != row_count:
        raise RuntimeError("synthetic evaluator panel identities are not unique")
    ordered_identity_digest = hashlib.sha256(
        json.dumps(identities, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    chains = torch.randn(
        row_count,
        config.flow.student_steps + 1,
        config.action.model_horizon,
        config.action.model_dimensions,
        generator=generator,
    )
    model_actions = chains[:, -1].detach().clone()
    contract = ActionContract(
        config.action.execution_prefix,
        config.action.physical_dimensions,
    )
    executed = contract.crop(model_actions)
    environment = _SyntheticEnvironment(config.action.physical_dimensions)
    recorded_rows: list[torch.Tensor] = []
    for row_actions in executed:
        environment.reset()
        for action in row_actions:
            environment.step(action)
        recorded_rows.append(environment.recorded_actions())
    recorded = torch.stack(recorded_rows)
    trace = validate_exact_trace(
        chains,
        model_actions,
        executed,
        recorded,
        expected_steps=config.flow.student_steps,
    )
    expected_shape = (
        config.action.execution_prefix,
        config.action.physical_dimensions,
    )
    if tuple(executed[0].shape) != expected_shape:
        raise RuntimeError(
            f"{benchmark} synthetic evaluator action shape drift: "
            f"{tuple(executed[0].shape)} != {expected_shape}"
        )
    return {
        "panel_rows": row_count,
        "unique_panel_identities": len(set(identities)),
        "ordered_identity_sha256": ordered_identity_digest,
        "solver_steps": config.flow.student_steps,
        "model_action_shape": list(model_actions.shape),
        "executed_action_shape": list(executed[0].shape),
        "recorded_action_shape": list(recorded[0].shape),
        "environment_steps_per_row": config.action.execution_prefix,
        "trace": trace,
    }


def run_evaluator_synthetic_smoke(
    benchmark: str,
    output_dir: Path,
    *,
    entrypoint: str,
    config: SyntheticSmokeConfig | None = None,
) -> dict[str, Any]:
    """Run a synthetic train/checkpoint/panel/action proof through an evaluator.

    This is intentionally a separate contract from a real benchmark evaluator:
    it uses the deterministic in-process policy and environment above, but
    still checks the same checkpoint, panel denominator, action crop, and exact
    trace boundaries that the public evaluator must preserve.
    """

    if benchmark not in PAPER_BENCHMARKS:
        raise ValueError(f"unsupported benchmark {benchmark!r}")
    config = config or SyntheticSmokeConfig()
    benchmark_config = load_benchmark_config(benchmark)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = output_dir / "training"
    training_report = run_benchmark_smoke(benchmark, training_dir, config)
    checkpoint = training_dir / f"{benchmark}-checkpoint.pt"
    if not checkpoint.is_file():
        raise RuntimeError("synthetic training did not produce an evaluator checkpoint")
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_payload.get("schema") != "path-opd-benchmark-synthetic-checkpoint-v1":
        raise RuntimeError("synthetic evaluator checkpoint schema mismatch")
    if checkpoint_payload.get("benchmark") != benchmark:
        raise RuntimeError("synthetic evaluator checkpoint benchmark mismatch")
    if checkpoint_payload.get("smoke_config") != asdict(config):
        raise RuntimeError("synthetic evaluator checkpoint config mismatch")
    if int(checkpoint_payload.get("next_update", -1)) != max(1, config.updates // 2):
        raise RuntimeError("synthetic evaluator checkpoint update mismatch")
    panel = _synthetic_panel(
        benchmark,
        benchmark_config,
        seed=config.seed,
        output_dir=output_dir,
    )
    panel_rows = panel.pop("_rows")
    action_trace = _synthetic_action_trace(
        benchmark,
        benchmark_config,
        seed=config.seed,
        panel_rows=panel_rows,
    )
    checks = {
        "checkpoint_schema": True,
        "checkpoint_reload": bool(training_report["checks"]["exact_resume"]),
        "panel_denominator": panel["rows"] == benchmark_config.evaluation.rows_per_model,
        "panel_unique_identities": panel["unique_identities"] == panel["rows"],
        "panel_action_rows": action_trace["panel_rows"] == panel["rows"],
        "panel_identity_binding": (
            action_trace["ordered_identity_sha256"] == panel["ordered_identity_sha256"]
            and action_trace["unique_panel_identities"] == panel["unique_identities"]
        ),
        "action_prefix_shape": action_trace["executed_action_shape"]
        == [
            benchmark_config.action.execution_prefix,
            benchmark_config.action.physical_dimensions,
        ],
        "exact_action_trace": bool(action_trace["trace"]["bitwise_executed_action"]),
        "exact_endpoint_trace": bool(action_trace["trace"]["bitwise_endpoint"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"synthetic evaluator smoke failed: {checks}")
    return {
        "schema": EVALUATOR_SCHEMA,
        "status": "PASS",
        "claim": "synthetic_evaluator_contract_smoke_only",
        "synthetic_only": True,
        "evaluator_entrypoint_executed": True,
        "entrypoint": entrypoint,
        "training_smoke_executed": True,
        "external_runtime_executed": False,
        "external_assets_used": False,
        "external_openpi_executed": False,
        "real_simulator": False,
        "gpu_executed": False,
        "paper_scale": False,
        "checkpoint": {
            "schema": checkpoint_payload["schema"],
            "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "next_update": checkpoint_payload["next_update"],
            "artifact": checkpoint.name,
        },
        "panel": panel,
        "action_trace": action_trace,
        "checks": checks,
        "training_report": training_report,
    }


def run_all_benchmark_smokes(
    output_dir: Path,
    config: SyntheticSmokeConfig | None = None,
) -> dict[str, Any]:
    """Run the same synthetic proof against all three paper configurations."""

    config = config or SyntheticSmokeConfig()
    reports = {
        benchmark: run_benchmark_smoke(benchmark, output_dir / benchmark, config)
        for benchmark in PAPER_BENCHMARKS
    }
    return {
        "schema": SCHEMA,
        "status": "PASS",
        **_evidence_boundary(),
        "benchmarks": reports,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write a stable JSON report for shell and CI consumers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

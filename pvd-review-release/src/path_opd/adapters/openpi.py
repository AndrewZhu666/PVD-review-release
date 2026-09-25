"""Thin adapter for an RLinf/OpenPI-compatible action model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from path_opd.core import ActionContract, FrozenTeacher, require_tensor


@dataclass(frozen=True)
class OpenPIAdapter:
    """Bind one rollout context to OpenPI NFT and DAgger calls.

    ``nft_forward_type`` and ``sft_forward_type`` are supplied by the host
    framework. Keeping them outside this package avoids vendoring RLinf solely
    for enum values.
    """

    model: nn.Module
    forward_inputs: Mapping[str, Any]
    steps: int
    nft_forward_type: Any
    sft_forward_type: Any | None = None
    context_keys: tuple[str, ...] | None = None
    contract: ActionContract | None = None

    @classmethod
    def from_rollout(
        cls,
        model: nn.Module,
        rollout_result: Mapping[str, Any],
        *,
        steps: int,
        nft_forward_type: Any,
        sft_forward_type: Any | None = None,
        context_keys: tuple[str, ...] | None = None,
        contract: ActionContract | None = None,
    ) -> OpenPIAdapter:
        if steps < 1:
            raise ValueError("OpenPI solver steps must be positive.")
        inputs = rollout_result.get("forward_inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("OpenPI rollout result lacks forward_inputs.")
        chains = require_tensor(inputs.get("chains"), "forward_inputs['chains']")
        if chains.ndim != 4 or chains.shape[1] != steps + 1:
            raise ValueError("OpenPI rollout chains do not match the configured solver steps.")
        return cls(
            model=model,
            forward_inputs=inputs,
            steps=steps,
            nft_forward_type=nft_forward_type,
            sft_forward_type=sft_forward_type,
            context_keys=context_keys,
            contract=contract,
        )

    @property
    def chains(self) -> torch.Tensor:
        return require_tensor(self.forward_inputs.get("chains"), "forward_inputs['chains']")

    def repeated_context(self) -> dict[str, Any]:
        """Repeat tensor observations once for every queried flow time."""
        keys = self.context_keys
        if keys is None:
            keys = tuple(
                key
                for key in self.forward_inputs
                if key.startswith("observation/")
                or key in {"tokenized_prompt", "tokenized_prompt_mask"}
            )
        context: dict[str, Any] = {}
        rollout_batch = self.chains.shape[0]
        for key in keys:
            value = self.forward_inputs.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"OpenPI context key {key!r} is missing or not a tensor.")
            if value.ndim == 0 or value.shape[0] != rollout_batch:
                raise ValueError(f"OpenPI context key {key!r} has the wrong batch dimension.")
            context[key] = value.repeat_interleave(self.steps, dim=0).detach().clone()
        return context

    def velocity(self, states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError("NFT states must have shape [batch, horizon, action_dim].")
        if times.ndim != 1 or times.shape[0] != states.shape[0]:
            raise ValueError("NFT times must be a vector aligned with the state batch.")
        output = self.model(
            forward_type=self.nft_forward_type,
            forward_inputs=self.repeated_context(),
            nft_inputs={"x_t": states, "timesteps": times},
            compute_values=False,
        )
        if not isinstance(output, Mapping) or "v_theta" not in output:
            raise RuntimeError("OpenPI NFT query did not return v_theta.")
        if "values" in output:
            raise RuntimeError("OpenPI NFT query leaked value outputs.")
        velocity = require_tensor(output["v_theta"], "OpenPI v_theta")
        if (
            velocity.ndim != 3
            or velocity.shape[0] != states.shape[0]
            or velocity.shape[-1] != states.shape[-1]
            or velocity.shape[-2] < 1
            or velocity.shape[-2] > states.shape[-2]
        ):
            raise ValueError(
                f"OpenPI velocity shape {tuple(velocity.shape)} is incompatible with "
                f"states {tuple(states.shape)}."
            )
        if self.contract is not None:
            self.contract.crop(velocity)
        if not torch.isfinite(velocity).all():
            raise FloatingPointError("OpenPI velocity contains NaN or Inf.")
        return velocity

    def frozen_teacher(self, identity: FrozenTeacher | None = None) -> FrozenTeacher:
        """Bind this rollout context to an immutable actor-only teacher.

        ``identity`` can be reused across updates; only the rollout query is
        rebound, so the expensive full-model fingerprint is captured once.
        """

        def query(model: nn.Module, states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
            return OpenPIAdapter(
                model=model,
                forward_inputs=self.forward_inputs,
                steps=self.steps,
                nft_forward_type=self.nft_forward_type,
                sft_forward_type=self.sft_forward_type,
                context_keys=self.context_keys,
                contract=self.contract,
            ).velocity(states, times)

        if identity is not None:
            if identity.model is not self.model:
                raise ValueError("Frozen teacher identity belongs to a different model.")
            return identity.bind(query)
        return FrozenTeacher(self.model, query)

    def endpoint_dagger_loss(self, teacher_model_actions: torch.Tensor) -> torch.Tensor:
        """Compute the native conditional flow-matching Endpoint-DAgger loss."""
        if self.sft_forward_type is None:
            raise RuntimeError("An SFT forward type is required for Endpoint-DAgger.")
        if teacher_model_actions.ndim < 2 or teacher_model_actions.shape[0] != self.chains.shape[0]:
            raise ValueError("Endpoint-DAgger target batch does not match rollout batch.")
        batch = dict(self.forward_inputs)
        batch["model_action"] = teacher_model_actions.reshape(
            teacher_model_actions.shape[0], -1
        ).detach()
        prepared = self.model.prepare_dagger_sft_batch(batch)
        loss = self.model(
            forward_type=self.sft_forward_type,
            data=prepared,
            use_action_chunk_loss=True,
        )
        return require_tensor(loss, "Endpoint-DAgger loss")

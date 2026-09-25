"""Contract tests for the optional RLinf/OpenPI integration boundary."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from path_opd.adapters.openpi import OpenPIAdapter


class _RecordingOpenPI(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.calls: list[dict[str, Any]] = []
        self.prepared_batches: list[dict[str, Any]] = []

    def prepare_dagger_sft_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.prepared_batches.append(batch)
        return {"prepared_target": batch["model_action"]}

    def forward(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if "nft_inputs" in kwargs:
            states = kwargs["nft_inputs"]["x_t"]
            times = kwargs["nft_inputs"]["timesteps"]
            return {"v_theta": states + times[:, None, None]}
        return kwargs["data"]["prepared_target"].square().mean() * self.anchor


def _adapter(model: nn.Module, *, steps: int = 2) -> OpenPIAdapter:
    rollout = {
        "forward_inputs": {
            "chains": torch.zeros(2, steps + 1, 1, 1),
            "observation/image": torch.tensor([[10.0], [20.0]], requires_grad=True),
            "observation/wrist_image": torch.tensor([[30.0], [40.0]]),
            "tokenized_prompt": torch.tensor([[1, 2], [3, 4]]),
            "denoise_inds": torch.tensor([0, 1]),
            "action": torch.full((2, 1), 40.0),
            "model_action": torch.full((2, 1), 50.0),
            "non_tensor_metadata": "not forwarded to NFT",
        }
    }
    return OpenPIAdapter.from_rollout(
        model,
        rollout,
        steps=steps,
        nft_forward_type="NFT",
        sft_forward_type="SFT",
    )


def test_nft_query_repeats_detached_context_and_disables_value_head() -> None:
    model = _RecordingOpenPI()
    adapter = _adapter(model)
    states = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    times = torch.tensor([1.0, 0.5, 1.0, 0.5])

    actual = adapter.velocity(states, times)

    torch.testing.assert_close(
        actual,
        torch.tensor([[[2.0]], [[2.5]], [[4.0]], [[4.5]]]),
        rtol=0.0,
        atol=0.0,
    )
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["forward_type"] == "NFT"
    assert call["compute_values"] is False
    assert call["nft_inputs"]["x_t"] is states
    assert call["nft_inputs"]["timesteps"] is times

    context = call["forward_inputs"]
    assert set(context) == {
        "observation/image",
        "observation/wrist_image",
        "tokenized_prompt",
    }
    torch.testing.assert_close(
        context["observation/image"],
        torch.tensor([[10.0], [10.0], [20.0], [20.0]]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        context["tokenized_prompt"],
        torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4]]),
        rtol=0.0,
        atol=0.0,
    )
    assert not context["observation/image"].requires_grad
    assert context["observation/image"].grad_fn is None


def test_nft_accepts_openpi_executable_prefix_velocity() -> None:
    class PrefixVelocity(_RecordingOpenPI):
        def forward(self, **kwargs: Any) -> Any:
            states = kwargs["nft_inputs"]["x_t"]
            return {"v_theta": states[:, :1, :]}

    model = PrefixVelocity()
    adapter = _adapter(model)
    states = torch.zeros(4, 2, 3)
    actual = adapter.velocity(states, torch.zeros(4))
    assert actual.shape == (4, 1, 3)


def test_endpoint_dagger_uses_native_sft_with_detached_flattened_target() -> None:
    model = _RecordingOpenPI()
    adapter = _adapter(model)
    teacher_actions = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        requires_grad=True,
    )

    loss = adapter.endpoint_dagger_loss(teacher_actions)
    loss.backward()

    assert len(model.prepared_batches) == 1
    target = model.prepared_batches[0]["model_action"]
    torch.testing.assert_close(
        target,
        torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        rtol=0.0,
        atol=0.0,
    )
    assert not target.requires_grad
    assert teacher_actions.grad is None
    assert model.anchor.grad is not None

    call = model.calls[0]
    assert call["forward_type"] == "SFT"
    assert call["use_action_chunk_loss"] is True
    assert set(call["data"]) == {"prepared_target"}


def test_nft_rejects_value_outputs_and_wrong_query_shapes() -> None:
    model = _RecordingOpenPI()
    adapter = _adapter(model)
    with torch.no_grad():
        model.calls.clear()

    with pytest.raises(ValueError, match="times"):
        adapter.velocity(torch.zeros(2, 1, 1), torch.zeros(1))

    original_forward = model.forward

    def leaking_forward(**kwargs: Any) -> Any:
        output = original_forward(**kwargs)
        if "nft_inputs" in kwargs:
            output["values"] = torch.zeros(kwargs["nft_inputs"]["x_t"].shape[0])
        return output

    model.forward = leaking_forward  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="leaked value"):
        adapter.velocity(torch.zeros(4, 1, 1), torch.zeros(4))


def test_from_rollout_rejects_chain_count_that_disagrees_with_solver() -> None:
    model = _RecordingOpenPI()
    rollout = {"forward_inputs": {"chains": torch.zeros(1, 4, 2, 3)}}

    try:
        OpenPIAdapter.from_rollout(
            model,
            rollout,
            steps=4,
            nft_forward_type="NFT",
        )
    except ValueError as error:
        assert "configured solver steps" in str(error)
    else:
        raise AssertionError("mismatched OpenPI chains were accepted")

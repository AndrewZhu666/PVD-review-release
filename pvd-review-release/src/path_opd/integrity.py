"""Deterministic integrity helpers for models and release artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_sha256(model: nn.Module, *, trainable_only: bool = False) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if trainable_only and not parameter.requires_grad:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        # ``Tensor.numpy`` would make NumPy an undeclared runtime dependency.
        # A byte view plus ``bytes`` is deterministic and stays within PyTorch
        # and the standard library, which is important for the clean CPU image.
        digest.update(bytes(value.reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()

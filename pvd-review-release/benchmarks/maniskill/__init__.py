"""Portable ManiSkill K8 benchmark contract and runtime helpers.

The benchmark modules deliberately keep RLinf, OpenPI, ManiSkill, and CUDA
imports lazy.  Contract inspection and ``--dry-run`` therefore work in a clean
CPU environment, while a real run fails closed unless the caller supplies all
external assets and an explicitly selected RLinf checkout.
"""

from .common import CONTRACT, ContractError, ResolvedContract, resolve_contract

__all__ = ["CONTRACT", "ContractError", "ResolvedContract", "resolve_contract"]

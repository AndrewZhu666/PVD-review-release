#!/usr/bin/env python3
"""Export the fixed ManiSkill reset panel from the selected RLinf runtime.

The exporter intentionally imports ManiSkill, Torch, and OmegaConf only after
the command line has been parsed.  A panel is therefore generated from the
same ``ManiskillEnv`` constructor used by the evaluator, rather than from a
copied RNG implementation or a checked-in list of IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    from .common import (
        CONTRACT,
        MANISKILL_CONTROL_MODE,
        ContractError,
        canonical_json_bytes,
        validate_rlinf_checkout,
    )
except ImportError:  # pragma: no cover - direct script invocation
    from common import (  # type: ignore[no-redef]
        CONTRACT,
        MANISKILL_CONTROL_MODE,
        ContractError,
        canonical_json_bytes,
        validate_rlinf_checkout,
    )

SCHEMA = "path-opd-maniskill-reset-panel-v1"
PINNED_RLINF_COMMIT = "fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c"
WORLD_SIZE = CONTRACT.world_size
ROWS_PER_RANK = CONTRACT.evaluation_rows // WORLD_SIZE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ManiSkill reset IDs from RLinf's ManiskillEnv"
    )
    parser.add_argument(
        "--rlinf-checkout",
        "--rlinf-root",
        dest="rlinf_checkout",
        type=Path,
        required=True,
        help="RLinf checkout containing rlinf/envs/maniskill/maniskill_env.py",
    )
    parser.add_argument(
        "--simulator-assets",
        type=Path,
        required=True,
        help="directory exported as MANISKILL_ASSET_DIR",
    )
    parser.add_argument(
        "--maniskill-package-assets",
        "--maniskill-assets",
        dest="maniskill_package_assets",
        type=Path,
        required=True,
        help="directory exported as MS_ASSET_DIR",
    )
    parser.add_argument("--output", type=Path, required=True, help="panel JSON destination")
    return parser


def _as_int_list(value: Any) -> list[int]:
    """Convert a runtime tensor/array to strict Python integer IDs."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ContractError("ManiskillEnv.reset_state_ids is not a sequence")
    result: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int or item < 0:
            raise ContractError(f"ManiskillEnv returned invalid reset ID at slot {index}")
        result.append(item)
    return result


def _close_environment(environment: Any) -> None:
    close = getattr(environment, "close", None)
    if callable(close):
        close()
        return
    nested = getattr(environment, "env", None)
    nested_close = getattr(nested, "close", None)
    if callable(nested_close):
        nested_close()


def _load_runtime(
    rlinf_checkout: Path,
    simulator_assets: Path,
    maniskill_package_assets: Path,
) -> tuple[Callable[..., Any], Callable[[int], Any]]:
    """Load the selected RLinf class and the evaluator's exact config builder."""

    checkout = validate_rlinf_checkout(rlinf_checkout, require_existing=True)
    try:
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not verify the selected RLinf checkout revision") from error
    if revision != PINNED_RLINF_COMMIT:
        raise RuntimeError(
            "RLinf checkout revision is not pinned: "
            f"expected {PINNED_RLINF_COMMIT}, got {revision}"
        )
    simulator_assets = simulator_assets.expanduser().resolve()
    maniskill_package_assets = maniskill_package_assets.expanduser().resolve()
    for label, path in (
        ("simulator-assets", simulator_assets),
        ("maniskill-package-assets", maniskill_package_assets),
    ):
        if not path.is_dir():
            raise ContractError(f"{label} is not a directory: {path}")
    os.environ["MANISKILL_ASSET_DIR"] = str(simulator_assets)
    os.environ["MS_ASSET_DIR"] = str(maniskill_package_assets)
    text = str(checkout)
    if not sys.path or sys.path[0] != text:
        sys.path.insert(0, text)
    importlib.invalidate_caches()
    try:
        loaded_rlinf = importlib.import_module("rlinf")
        from omegaconf import OmegaConf
        from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
    except ImportError as error:  # pragma: no cover - depends on external runtime
        raise RuntimeError(
            "ManiSkill panel export requires the selected RLinf checkout, "
            "OmegaConf, Torch, and ManiSkill"
        ) from error
    loaded_path = getattr(loaded_rlinf, "__file__", None)
    try:
        Path(loaded_path).resolve().relative_to(checkout)
    except (TypeError, ValueError):
        raise RuntimeError("selected RLinf checkout did not win import resolution") from None

    try:
        from .evaluate import _environment_config
    except ImportError:  # pragma: no cover - direct script invocation
        from evaluate import _environment_config  # type: ignore[no-redef]

    return ManiskillEnv, lambda rows: _environment_config(OmegaConf, rows)


def collect_reset_rows(
    rlinf_checkout: Path,
    simulator_assets: Path,
    maniskill_package_assets: Path,
    *,
    env_factory: Callable[..., Any] | None = None,
    config_factory: Callable[[int], Any] | None = None,
) -> list[dict[str, int]]:
    """Create one fixed runtime per rank and collect its real reset IDs.

    ``env_factory`` and ``config_factory`` are dependency-free test seams.  In
    normal use they are omitted and the function imports the pinned RLinf
    ``ManiskillEnv`` and the release evaluator's configuration.
    """

    if env_factory is None or config_factory is None:
        loaded_factory, loaded_config = _load_runtime(
            rlinf_checkout, simulator_assets, maniskill_package_assets
        )
        env_factory = loaded_factory
        config_factory = loaded_config

    rows: list[dict[str, int]] = []
    all_ids: list[int] = []
    for rank in range(WORLD_SIZE):
        cfg = config_factory(ROWS_PER_RANK)
        environment = None
        try:
            environment = env_factory(
                cfg=cfg,
                num_envs=ROWS_PER_RANK,
                seed_offset=rank,
                total_num_processes=WORLD_SIZE,
                worker_info=None,
                record_metrics=False,
            )
            reset = getattr(environment, "reset", None)
            if not callable(reset):
                raise ContractError("ManiskillEnv does not expose reset()")
            reset()
            ids = _as_int_list(getattr(environment, "reset_state_ids", None))
            if len(ids) != ROWS_PER_RANK:
                raise ContractError(
                    f"rank {rank} returned {len(ids)} reset IDs; expected {ROWS_PER_RANK}"
                )
            if len(set(ids)) != len(ids):
                raise ContractError(f"rank {rank} returned duplicate reset IDs")
            rows.extend(
                {
                    "reset_episode_id": reset_id,
                    "initial_state_id": reset_id,
                    "worker_rank": rank,
                    "pipeline_stage": 0,
                    "env_slot": slot,
                }
                for slot, reset_id in enumerate(ids)
            )
            all_ids.extend(ids)
        finally:
            if environment is not None:
                _close_environment(environment)

    if len(rows) != CONTRACT.evaluation_rows:
        raise ContractError(f"collected {len(rows)} rows; expected {CONTRACT.evaluation_rows}")
    if len(set(all_ids)) != len(all_ids):
        raise ContractError("ManiskillEnv returned duplicate reset IDs across ranks")
    return rows


def _panel_payload(rows: list[dict[str, int]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "environment_id": CONTRACT.task,
        "denominator": CONTRACT.evaluation_rows,
        "seed": 0,
        "world_size": WORLD_SIZE,
        "rows_per_rank": ROWS_PER_RANK,
        "control_mode": MANISKILL_CONTROL_MODE,
        "source": {
            "api": "rlinf.envs.maniskill.maniskill_env.ManiskillEnv",
            "reset_field": "reset_state_ids",
            "seed_rule": "cfg.seed + seed_offset",
        },
        "rows": rows,
    }
    payload["ordered_reset_ids_sha256"] = hashlib.sha256(
        canonical_json_bytes([row["reset_episode_id"] for row in rows])
    ).hexdigest()
    return payload


def _write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return hashlib.sha256(data).hexdigest()


def export_panel(
    *,
    rlinf_checkout: Path,
    simulator_assets: Path,
    maniskill_package_assets: Path,
    output: Path,
    env_factory: Callable[..., Any] | None = None,
    config_factory: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    rows = collect_reset_rows(
        rlinf_checkout,
        simulator_assets,
        maniskill_package_assets,
        env_factory=env_factory,
        config_factory=config_factory,
    )
    payload = _panel_payload(rows)
    content = dict(payload)
    content_sha256 = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    payload["panel_sha256"] = content_sha256
    panel_bytes = canonical_json_bytes(payload)
    panel_sha256 = _write_bytes(output, panel_bytes)
    rows_path = output.with_name(f"{output.stem}.rows.jsonl")
    rows_bytes = b"".join(canonical_json_bytes(row) for row in rows)
    rows_sha256 = _write_bytes(rows_path, rows_bytes)
    return {
        "schema": SCHEMA,
        "benchmark": "maniskill",
        "panel_path": str(output),
        "panel_sha256": panel_sha256,
        "content_sha256": content_sha256,
        "rows_jsonl_path": str(rows_path),
        "rows_jsonl_sha256": rows_sha256,
        "row_count": len(rows),
        "ordered_reset_ids_sha256": payload["ordered_reset_ids_sha256"],
        "source_api": payload["source"]["api"],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = export_panel(
            rlinf_checkout=args.rlinf_checkout,
            simulator_assets=args.simulator_assets,
            maniskill_package_assets=args.maniskill_package_assets,
            output=args.output,
        )
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"ManiSkill panel export failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export the Official-D CALVIN sequence panel from RLinf's upstream API.

The sequence generator is deliberately not reimplemented here.  The selected
RLinf checkout owns the import of ``get_sequences`` so that its pinned
CALVIN dependencies, seeding, filtering, and shuffle behavior remain the
source of the exported rows.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .common import (
        PROTOCOL,
        ContractError,
        activate_rlinf,
        canonical_json,
        sha256_file,
        validate_checkout,
    )
except ImportError:  # pragma: no cover - direct script invocation
    from common import (  # type: ignore[no-redef]
        PROTOCOL,
        ContractError,
        activate_rlinf,
        canonical_json,
        sha256_file,
        validate_checkout,
    )

SCHEMA = "fm-opd-calvin-official-d-sequence-panel-v1"
PINNED_RLINF_COMMIT = "fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c"
CALVIN_COMMIT = "fa03f01f19c65920e18cf37398a9ce859274af76"
CALVIN_SEQUENCE_SOURCE_SHA256 = "fbd08f8501b1c96ce564b085881e31a48396e9ab1aebdda1e75570e3387fad29"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export CALVIN Official-D sequences from RLinf's get_sequences API"
    )
    parser.add_argument(
        "--rlinf-checkout",
        "--rlinf-root",
        dest="rlinf_checkout",
        type=Path,
        required=True,
        help="RLinf checkout exposing rlinf.envs.calvin.utils.get_sequences",
    )
    parser.add_argument("--output", type=Path, required=True, help="panel JSON destination")
    return parser


def _normalize(value: Any) -> Any:
    """Convert upstream numpy/container values to deterministic JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _normalize(value.item())
    if hasattr(value, "tolist"):
        return _normalize(value.tolist())
    raise ContractError(f"CALVIN sequence contains a non-JSON value: {type(value).__name__}")


def _load_get_sequences(rlinf_checkout: Path) -> Callable[[int], Any]:
    checkout = validate_checkout(rlinf_checkout, required=True)
    utils_path = checkout / "rlinf/envs/calvin/utils.py"
    if not utils_path.is_file():
        raise ContractError(f"rlinf-checkout is missing: {utils_path.relative_to(checkout)}")
    try:
        rlinf_revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not verify the selected RLinf checkout revision") from error
    if rlinf_revision != PINNED_RLINF_COMMIT:
        raise RuntimeError(
            "RLinf checkout revision is not pinned: "
            f"expected {PINNED_RLINF_COMMIT}, got {rlinf_revision}"
        )
    try:
        activate_rlinf(checkout)
        from calvin_agent.evaluation.multistep_sequences import get_sequences_for_state2
        from rlinf.envs.calvin.utils import get_sequences
    except ImportError as error:  # pragma: no cover - depends on external runtime
        raise RuntimeError(
            "CALVIN panel export requires the selected RLinf checkout and its "
            "CALVIN/calvin_agent dependencies"
        ) from error
    sequence_source = inspect.getsourcefile(get_sequences_for_state2)
    if sequence_source is None:
        raise RuntimeError("could not identify the imported CALVIN sequence source file")
    source_path = Path(sequence_source).resolve()
    if sha256_file(source_path) != CALVIN_SEQUENCE_SOURCE_SHA256:
        raise RuntimeError(
            "CALVIN sequence source does not match the pinned upstream implementation "
            f"{CALVIN_COMMIT}"
        )
    try:
        revision = subprocess.run(
            ["git", "-C", str(source_path.parents[3]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not verify the CALVIN source checkout revision") from error
    if revision != CALVIN_COMMIT:
        raise RuntimeError(
            f"CALVIN checkout revision is not pinned: expected {CALVIN_COMMIT}, got {revision}"
        )
    return get_sequences


def collect_rows(
    rlinf_checkout: Path,
    *,
    sequence_factory: Callable[[int], Any] | None = None,
) -> list[dict[str, Any]]:
    """Call the fixed upstream API and validate its complete 1,000-row result."""

    factory = (
        sequence_factory
        if sequence_factory is not None
        else _load_get_sequences(rlinf_checkout)
    )
    if PROTOCOL.evaluation_rows != 1000:
        raise ContractError("CALVIN protocol no longer requests get_sequences(1000)")
    try:
        generated = factory(1000)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as error:
        raise RuntimeError(f"RLinf get_sequences(1000) failed: {error}") from error
    if not isinstance(generated, Sequence) or isinstance(generated, (str, bytes)):
        raise ContractError("RLinf get_sequences(1000) did not return a sequence")
    if len(generated) != PROTOCOL.evaluation_rows:
        raise ContractError(
            f"RLinf get_sequences(1000) returned {len(generated)} rows; "
            f"expected {PROTOCOL.evaluation_rows}"
        )

    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    semantic_identities: set[str] = set()
    for index, item in enumerate(generated):
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise ContractError(f"CALVIN sequence {index} must be (initial_state, subtasks)")
        initial_state = _normalize(item[0])
        subtasks = _normalize(item[1])
        if not isinstance(initial_state, Mapping):
            raise ContractError(f"CALVIN sequence {index} initial_state is not a mapping")
        if (
            not isinstance(subtasks, list)
            or len(subtasks) != PROTOCOL.sequence_length
            or not all(isinstance(task, str) and task for task in subtasks)
        ):
            raise ContractError(
                f"CALVIN sequence {index} must contain "
                f"{PROTOCOL.sequence_length} non-empty subtasks"
            )
        identity_payload = {
            "sequence_index": index,
            "initial_state": dict(initial_state),
            "subtasks": subtasks,
        }
        semantic_identity = hashlib.sha256(
            canonical_json(
                {"initial_state": dict(initial_state), "subtasks": subtasks}
            )
        ).hexdigest()
        if semantic_identity in semantic_identities:
            raise ContractError(f"CALVIN sequence {index} duplicates an earlier sequence")
        semantic_identities.add(semantic_identity)
        identity = hashlib.sha256(canonical_json(identity_payload)).hexdigest()
        if identity in identities:
            raise ContractError(f"CALVIN sequence {index} duplicates an earlier identity")
        identities.add(identity)
        rows.append({**identity_payload, "identity_sha256": identity})
    return rows


def _write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return hashlib.sha256(data).hexdigest()


def export_panel(
    *,
    rlinf_checkout: Path,
    output: Path,
    sequence_factory: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    rows = collect_rows(rlinf_checkout, sequence_factory=sequence_factory)
    ordered_identity_sha256 = hashlib.sha256(
        "\n".join(str(row["identity_sha256"]) for row in rows).encode("utf-8")
    ).hexdigest()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "source": "CALVIN official get_sequences(1000)",
        "calvin_commit": CALVIN_COMMIT,
        "sequence_source_sha256": CALVIN_SEQUENCE_SOURCE_SHA256,
        "sequence_count": PROTOCOL.evaluation_rows,
        "subtasks_per_sequence": PROTOCOL.sequence_length,
        "ordered_identity_sha256": ordered_identity_sha256,
        "rows": rows,
    }
    panel_bytes = canonical_json(payload)
    panel_sha256 = _write_bytes(output, panel_bytes)
    rows_path = output.with_name(f"{output.stem}.rows.jsonl")
    rows_bytes = b"".join(canonical_json(row) for row in rows)
    rows_sha256 = _write_bytes(rows_path, rows_bytes)
    return {
        "schema": SCHEMA,
        "benchmark": PROTOCOL.benchmark,
        "panel_path": str(output),
        "panel_sha256": panel_sha256,
        "content_sha256": panel_sha256,
        "rows_jsonl_path": str(rows_path),
        "rows_jsonl_sha256": rows_sha256,
        "row_count": len(rows),
        "ordered_identity_sha256": ordered_identity_sha256,
        "source_api": "rlinf.envs.calvin.utils.get_sequences",
        "source_call": "get_sequences(1000)",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = export_panel(rlinf_checkout=args.rlinf_checkout, output=args.output)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"CALVIN panel export failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

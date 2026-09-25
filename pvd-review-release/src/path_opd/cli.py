"""Command-line entry points for release verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from path_opd.adapters.benchmark_smoke import (
    SyntheticSmokeConfig,
    run_all_benchmark_smokes,
    run_benchmark_smoke,
    write_report,
)
from path_opd.adapters.toy import ToyRunConfig, run_smoke
from path_opd.assets import AssetIntegrityError, AssetResolutionError
from path_opd.benchmark import (
    METHODS,
    PURPOSES,
    describe_benchmark,
    parse_asset_assignments,
    preflight_benchmark,
)
from path_opd.config import DEFAULT_CONFIG_DIRECTORY, PAPER_BENCHMARKS, ConfigError


def _write_json(path: Path | None, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(rendered, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    print(path)


def _smoke(args: argparse.Namespace) -> int:
    config = ToyRunConfig(updates=args.updates, seed=args.seed)
    result = run_smoke(args.work_dir, config)
    _write_json(args.output, result)
    return 0


def _benchmark_describe(args: argparse.Namespace) -> int:
    result = describe_benchmark(
        args.benchmark,
        config_directory=args.config_directory,
    )
    _write_json(args.output, result)
    return 0


def _benchmark_preflight(args: argparse.Namespace) -> int:
    result = preflight_benchmark(
        args.benchmark,
        purpose=args.purpose,
        supplied=parse_asset_assignments(args.asset),
        method=args.method,
        seed=args.seed,
        config_directory=args.config_directory,
    )
    _write_json(args.output, result)
    return 0


def _benchmark_smoke(args: argparse.Namespace) -> int:
    config = SyntheticSmokeConfig(updates=args.updates, batch_size=args.batch_size, seed=args.seed)
    if args.benchmark == "all":
        result = run_all_benchmark_smokes(args.work_dir, config)
    else:
        result = run_benchmark_smoke(args.benchmark, args.work_dir / args.benchmark, config)
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True) + "\n", end="")
    else:
        write_report(args.output, result)
        print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="path-opd")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="run deterministic CPU Path-OPD proof")
    smoke.add_argument("--work-dir", type=Path, default=Path("artifacts/toy-smoke"))
    smoke.add_argument("--output", type=Path)
    smoke.add_argument("--updates", type=int, default=40)
    smoke.add_argument("--seed", type=int, default=7)
    smoke.set_defaults(handler=_smoke)

    benchmark = commands.add_parser(
        "benchmark",
        help="inspect benchmark contracts and verify supplied assets",
    )
    benchmark_commands = benchmark.add_subparsers(
        dest="benchmark_command",
        required=True,
    )
    describe = benchmark_commands.add_parser(
        "describe",
        help="print one validated paper benchmark contract",
    )
    describe.add_argument("--benchmark", choices=PAPER_BENCHMARKS, required=True)
    describe.add_argument(
        "--config-directory",
        type=Path,
        default=DEFAULT_CONFIG_DIRECTORY,
    )
    describe.add_argument("--output", type=Path)
    describe.set_defaults(handler=_benchmark_describe)

    preflight = benchmark_commands.add_parser(
        "preflight",
        help="resolve and hash-check assets for one operation",
    )
    preflight.add_argument("--benchmark", choices=PAPER_BENCHMARKS, required=True)
    preflight.add_argument("--purpose", choices=PURPOSES, required=True)
    preflight.add_argument("--method", choices=METHODS)
    preflight.add_argument("--seed", type=int)
    preflight.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="supply one manifest parameter; repeat for every required asset",
    )
    preflight.add_argument(
        "--config-directory",
        type=Path,
        default=DEFAULT_CONFIG_DIRECTORY,
    )
    preflight.add_argument("--output", type=Path)
    preflight.set_defaults(handler=_benchmark_preflight)
    synthetic_smoke = benchmark_commands.add_parser(
        "smoke",
        help="run a synthetic CPU contract check (not real benchmark qualification)",
    )
    synthetic_smoke.add_argument(
        "--benchmark",
        choices=(*PAPER_BENCHMARKS, "all"),
        default="all",
    )
    synthetic_smoke.add_argument("--work-dir", type=Path, default=Path("artifacts/benchmark-smoke"))
    synthetic_smoke.add_argument("--output", type=Path)
    synthetic_smoke.add_argument("--updates", type=int, default=4)
    synthetic_smoke.add_argument("--batch-size", type=int, default=2)
    synthetic_smoke.add_argument("--seed", type=int, default=7)
    synthetic_smoke.set_defaults(handler=_benchmark_smoke)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except (
        AssetIntegrityError,
        AssetResolutionError,
        ConfigError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"path-opd: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

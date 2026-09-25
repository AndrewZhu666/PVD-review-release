from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import path_opd.benchmark as benchmark_module  # noqa: E402
from path_opd.adapters import benchmark_smoke as benchmark_smoke_module  # noqa: E402
from path_opd.adapters.benchmark_smoke import (  # noqa: E402
    SyntheticSmokeConfig,
    run_all_benchmark_smokes,
    run_benchmark_smoke,
)
from path_opd.assets import (  # noqa: E402
    Asset,
    AssetIntegrity,
    AssetLicense,
    AssetLocation,
    AssetManifest,
    load_asset_manifest,
)
from path_opd.benchmark import (  # noqa: E402
    parse_asset_assignments,
    preflight_benchmark,
    select_assets,
)
from path_opd.cli import build_parser, main  # noqa: E402
from path_opd.config import (  # noqa: E402
    DEFAULT_CONFIG_DIRECTORY,
    ConfigError,
    load_benchmark_config,
)


def _ids(manifest: AssetManifest) -> set[str]:
    return {asset.id for asset in manifest.assets}


def _asset(
    asset_id: str,
    parameter: str,
    *,
    sha256: str | None,
    required: bool = True,
) -> Asset:
    return Asset(
        id=asset_id,
        kind="model",
        description=f"Test asset {asset_id}",
        required=required,
        used_by=("train",),
        location=AssetLocation(
            type="user_parameter",
            parameter=parameter,
            relative_path=None,
            public_url_status="unknown",
            public_url=None,
        ),
        license=AssetLicense(status="unknown", identifier=None),
        integrity=AssetIntegrity(
            status="known" if sha256 is not None else "unknown",
            algorithm="sha256_file",
            sha256=sha256,
        ),
        expected_filename=None,
        selector=None,
    )


def test_asset_selection_matches_each_operation() -> None:
    config = load_benchmark_config("maniskill")
    manifest = load_asset_manifest("maniskill")

    training = select_assets(config, manifest, purpose="train")
    evaluation = select_assets(config, manifest, purpose="evaluate")
    reproduction = select_assets(
        config,
        manifest,
        purpose="reproduce_paper_results",
        method="path_opd",
        seed=1,
    )

    assert _ids(training) == {
        "base_model",
        "frozen_teacher",
        "normalization_stats",
        "teacher_normalization_stats",
        "simulator_assets",
        "package_assets",
    }
    assert _ids(evaluation) == {
        "base_model",
        "normalization_stats",
        "simulator_assets",
        "package_assets",
        "evaluation_panel",
    }
    assert "frozen_teacher" not in _ids(reproduction)
    assert "checkpoint_path_opd_seed_1" in _ids(reproduction)
    assert not any(
        asset.selector is not None
        and (asset.selector.method != "path_opd" or asset.selector.training_seed != 1)
        for asset in reproduction.assets
    )


@pytest.mark.parametrize(
    ("method", "seed"),
    [("path_opd", None), (None, 0)],
)
def test_asset_selection_requires_method_and_seed_together(
    method: str | None,
    seed: int | None,
) -> None:
    config = load_benchmark_config("maniskill")
    manifest = load_asset_manifest("maniskill")

    with pytest.raises(ConfigError, match="must be supplied together"):
        select_assets(
            config,
            manifest,
            purpose="reproduce_paper_results",
            method=method,
            seed=seed,
        )


def test_non_reproduction_preflight_rejects_checkpoint_selector() -> None:
    config = load_benchmark_config("maniskill")
    manifest = load_asset_manifest("maniskill")

    with pytest.raises(ConfigError, match="sealed paper checkpoints"):
        select_assets(
            config,
            manifest,
            purpose="evaluate",
            method="path_opd",
            seed=0,
        )


def test_preflight_reports_known_and_unknown_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known = tmp_path / "known.bin"
    unknown = tmp_path / "unknown.bin"
    known.write_bytes(b"known bytes")
    unknown.write_bytes(b"unknown bytes")
    digest = hashlib.sha256(known.read_bytes()).hexdigest()
    manifest = AssetManifest(
        schema_version=1,
        benchmark="maniskill",
        assets=(
            _asset("known_asset", "known_path", sha256=digest),
            _asset("unknown_asset", "unknown_path", sha256=None),
        ),
    )
    monkeypatch.setattr(benchmark_module, "load_asset_manifest", lambda *args, **kwargs: manifest)

    report = preflight_benchmark(
        "maniskill",
        purpose="train",
        supplied={"known_path": known, "unknown_path": unknown},
    )

    assert report["status"] == "PASS_WITH_UNVERIFIED_ASSETS"
    assert report["unverified_assets"] == ["unknown_asset"]
    assert {entry["id"]: entry["integrity"] for entry in report["assets"]} == {
        "known_asset": "verified",
        "unknown_asset": "unknown",
    }
    assert report["warning"] is not None


def test_preflight_allows_an_unsupplied_optional_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required = tmp_path / "required.bin"
    required.write_bytes(b"required")
    manifest = AssetManifest(
        schema_version=1,
        benchmark="maniskill",
        assets=(
            _asset("required_asset", "required_path", sha256=None),
            _asset("optional_asset", "optional_path", sha256=None, required=False),
        ),
    )
    monkeypatch.setattr(benchmark_module, "load_asset_manifest", lambda *args, **kwargs: manifest)

    report = preflight_benchmark(
        "maniskill",
        purpose="train",
        supplied={"required_path": required},
    )

    by_id = {entry["id"]: entry for entry in report["assets"]}
    assert by_id["optional_asset"]["path"] is None
    assert by_id["optional_asset"]["integrity"] == "not_supplied"
    assert report["unverified_assets"] == ["required_asset"]


def test_parse_asset_assignments_rejects_ambiguity() -> None:
    assert parse_asset_assignments(["base=/tmp/base", "panel=/tmp/panel"]) == {
        "base": Path("/tmp/base"),
        "panel": Path("/tmp/panel"),
    }
    with pytest.raises(ConfigError, match="duplicate"):
        parse_asset_assignments(["base=/tmp/a", "base=/tmp/b"])
    with pytest.raises(ConfigError, match="NAME=PATH"):
        parse_asset_assignments(["missing-separator"])
    with pytest.raises(ConfigError, match="lower_snake_case"):
        parse_asset_assignments(["Not-Snake=/tmp/value"])


def test_cli_default_config_is_independent_of_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parsed = build_parser().parse_args(["benchmark", "describe", "--benchmark", "metaworld_mt50"])
    assert parsed.config_directory == DEFAULT_CONFIG_DIRECTORY
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["path-opd", "benchmark", "describe", "--benchmark", "metaworld_mt50"],
    )

    assert main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["benchmark"]["benchmark"] == "metaworld_mt50"
    assert output["benchmark"]["flow"]["student_steps"] == 5


def test_synthetic_smoke_executes_all_three_configured_contracts(tmp_path: Path) -> None:
    report = run_all_benchmark_smokes(
        tmp_path,
        config=SyntheticSmokeConfig(updates=2, batch_size=1, seed=13),
    )

    assert report["schema"] == "path-opd-benchmark-synthetic-smoke-v2"
    assert report["status"] == "PASS"
    assert report["claim"] == "synthetic_contract_smoke_only"
    assert report["synthetic_only"] is True
    assert report["path_opd_core_executed"] is True
    assert report["checkpoint_reload_executed"] is True
    assert report["external_assets_used"] is False
    assert report["openpi_adapter_executed"] is True
    assert report["synthetic_openpi_adapter_executed"] is True
    assert report["external_openpi_executed"] is False
    assert report["endpoint_dagger_executed"] is True
    assert report["synthetic_endpoint_dagger_executed"] is True
    assert report["external_endpoint_dagger_executed"] is False
    assert report["real_simulator"] is False
    assert report["gpu_executed"] is False
    assert report["paper_scale"] is False
    assert set(report["benchmarks"]) == {"maniskill", "calvin_abc_d", "metaworld_mt50"}
    expected_contracts = {
        "maniskill": (8, 8, 5, 7),
        "calvin_abc_d": (8, 5, 5, 7),
        "metaworld_mt50": (5, 5, 5, 4),
    }
    for benchmark, payload in report["benchmarks"].items():
        assert payload["status"] == "PASS", benchmark
        assert tuple(payload["contract"].values()) == expected_contracts[benchmark]
        assert payload["uninterrupted"]["environment"]["steps"] == payload["contract"][
            "execution_prefix"
        ]
        assert payload["uninterrupted"]["environment"]["received_action_count"] == payload[
            "contract"
        ]["execution_prefix"]
        assert (
            payload["uninterrupted"]["environment"]["expected_action_sha256"]
            == payload["uninterrupted"]["environment"]["received_action_sha256"]
        )
        assert payload["checks"] == {
            "loss_decreased": True,
            "teacher_unchanged": True,
            "endpoint_contract": True,
            "physical_action_contract": True,
            "exact_model_resume": True,
            "exact_optimizer_resume": True,
            "exact_generator_resume": True,
            "exact_resume": True,
            "endpoint_dagger_loss_finite": True,
            "endpoint_dagger_target_detached": True,
            "endpoint_dagger_student_gradient": True,
        }
        assert payload["endpoint_dagger"]["loss_finite"] is True
        assert payload["endpoint_dagger"]["target_detached"] is True
        assert payload["endpoint_dagger"]["student_gradient"] is True


def test_synthetic_smoke_validates_actions_received_by_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MutatingEnvironment(benchmark_smoke_module._SyntheticEnvironment):
        def step(self, action):
            super().step(action + 1)

    monkeypatch.setattr(benchmark_smoke_module, "_SyntheticEnvironment", _MutatingEnvironment)
    with pytest.raises(RuntimeError, match="Exact trace provenance failed"):
        run_benchmark_smoke(
            "maniskill",
            tmp_path,
            config=SyntheticSmokeConfig(updates=2, batch_size=1, seed=13),
        )

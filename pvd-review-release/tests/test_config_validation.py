from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from path_opd.assets import (  # noqa: E402
    AssetIntegrityError,
    AssetResolutionError,
    load_all_asset_manifests,
    load_asset_manifest,
    resolve_asset_paths,
    validate_asset_payload,
    verify_asset_integrity,
)
from path_opd.config import (  # noqa: E402
    ConfigError,
    load_benchmark_config,
    validate_benchmark_payload,
)


def _benchmark_payload() -> dict[str, object]:
    path = REPOSITORY_ROOT / "configs" / "maniskill.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _asset_payload() -> dict[str, object]:
    path = REPOSITORY_ROOT / "configs" / "assets" / "maniskill.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_benchmark_loader_rejects_unknown_names() -> None:
    with pytest.raises(ConfigError, match="unsupported benchmark"):
        load_benchmark_config("not_a_paper_benchmark")


def test_benchmark_validator_rejects_unknown_fields() -> None:
    payload = _benchmark_payload()
    payload["private_extension"] = True

    with pytest.raises(ConfigError, match="unexpected field"):
        validate_benchmark_payload(payload)


def test_benchmark_validator_checks_budget_relationships() -> None:
    payload = _benchmark_payload()
    training = payload["training"]
    assert isinstance(training, dict)
    training["global_batch_size"] = 16

    with pytest.raises(ConfigError, match="global_batch_size"):
        validate_benchmark_payload(payload)


def test_benchmark_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    (config_directory / "maniskill.json").write_text(
        '{"schema_version": 1, "schema_version": 1}', encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="duplicate JSON key"):
        load_benchmark_config("maniskill", config_directory=config_directory)


def test_benchmark_config_references_its_asset_manifest() -> None:
    for benchmark in ("calvin_abc_d", "maniskill", "metaworld_mt50"):
        config = load_benchmark_config(benchmark)
        assert config.assets_manifest == f"assets/{benchmark}.json"
        manifest = load_asset_manifest(benchmark)
        if config.evaluation.panel_asset_id is not None:
            assert any(asset.id == config.evaluation.panel_asset_id for asset in manifest.assets)


def test_asset_manifests_cover_exactly_the_paper_benchmarks() -> None:
    manifests = load_all_asset_manifests()

    assert set(manifests) == {"calvin_abc_d", "maniskill", "metaworld_mt50"}
    for benchmark, manifest in manifests.items():
        assert manifest.benchmark == benchmark
        assert manifest.assets


def test_asset_routes_and_unknowns_are_explicit() -> None:
    known_routes = {
        ("maniskill", "simulator_assets"): {
            "url": "https://huggingface.co/datasets/RLinf/maniskill_assets",
            "revision": "c23fc1880ed7861686d4f995360101eaee4d18a0",
            "license": None,
        },
            ("maniskill", "base_model"): {
                "url": "https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT",
                "revision": "eb0e2c90726f7f0bdf99e8c9b6656917f0eb5030",
                "license": None,
            },
            ("maniskill", "frozen_teacher"): {
                "url": "https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE",
                "revision": "a791196be19a4d35094e1e7eaf4984da3a0ec5ba",
                "artifact_path": "actor/model.safetensors",
                "license": None,
            },
            ("maniskill", "normalization_stats"): {
                "url": "https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT",
                "revision": "eb0e2c90726f7f0bdf99e8c9b6656917f0eb5030",
                "artifact_path": "physical-intelligence/maniskill/norm_stats.json",
                "license": None,
            },
            ("maniskill", "teacher_normalization_stats"): {
                "url": "https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE",
                "revision": "a791196be19a4d35094e1e7eaf4984da3a0ec5ba",
                "artifact_path": "actor/assets/global_step_150/meta/norm_stats.json",
                "license": None,
            },
        ("calvin_abc_d", "base_model"): {
            "url": "https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT",
            "revision": "ffe18938a37d3855c9ff50d6220d39bcd6434f15",
            "license": "Apache-2.0",
        },
        ("calvin_abc_d", "normalization_stats"): {
            "url": "https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT",
            "revision": "ffe18938a37d3855c9ff50d6220d39bcd6434f15",
            "license": "Apache-2.0",
        },
        ("calvin_abc_d", "frozen_teacher"): {
            "url": "https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowSDE",
            "revision": "712bd3da6f8a92be8004ccf53078af70cc945bc7",
            "license": None,
        },
            ("metaworld_mt50", "base_model"): {
                "url": "https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT",
                "revision": "60b2d0309af8bb087c9e9b250f8c7f9db50c0f8e",
                "license": None,
            },
            ("metaworld_mt50", "frozen_teacher"): {
                "url": "https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowSDE",
                "revision": "22a30b31e1144a47bf2dc60bcb70c03c5abdf036",
                "license": None,
            },
            ("metaworld_mt50", "normalization_stats"): {
                "url": "https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT",
                "revision": "60b2d0309af8bb087c9e9b250f8c7f9db50c0f8e",
                "artifact_path": "lerobot/metaworld_mt50/norm_stats.json",
                "license": None,
            },
    }
    for manifest in load_all_asset_manifests().values():
        for asset in manifest.assets:
            assert asset.required is True
            assert asset.location.type == "user_parameter"
            assert asset.location.parameter
            assert asset.location.relative_path is None
            expected = known_routes.get((manifest.benchmark, asset.id))
            if expected is None:
                assert asset.location.public_url_status == "unknown"
                assert asset.location.public_url is None
                assert asset.location.revision_status == "unknown"
                assert asset.location.revision is None
                assert asset.license.status == "unknown"
                assert asset.license.identifier is None
                continue
            assert asset.location.public_url_status == "known"
            assert asset.location.public_url == expected["url"]
            if expected["revision"] is None:
                assert asset.location.revision_status == "unknown"
                assert asset.location.revision is None
            else:
                assert asset.location.revision_status == "known"
                assert asset.location.revision == expected["revision"]
            if "artifact_path" in expected:
                assert asset.location.artifact_path == expected["artifact_path"]
            if expected["license"] is None:
                assert asset.license.status == "unknown"
                assert asset.license.identifier is None
            else:
                assert asset.license.status == "known"
                assert asset.license.identifier == expected["license"]


def test_calvin_normalizer_declares_exact_public_artifact_path() -> None:
    asset = next(
        asset
        for asset in load_asset_manifest("calvin_abc_d").assets
        if asset.id == "normalization_stats"
    )
    assert asset.location.artifact_path == "InternRobotics/InternData-Calvin_ABC/norm_stats.json"
    assert asset.expected_filename == "norm_stats.json"


def test_known_public_routes_retain_recorded_hashes() -> None:
    manifests = load_all_asset_manifests()
    expected_hashes = {
        ("maniskill", "simulator_assets"): None,
        ("calvin_abc_d", "base_model"):
            "1ac4fcd76dfe131f9ce4cec94a750424f9eac0275285ebda7ad05ee9550c3bf8",
        ("calvin_abc_d", "normalization_stats"):
            "72903e3dad8fc7f4f799e36ee723d0c7f53cb815cc6c55f8ad75f62358f00b41",
        ("calvin_abc_d", "frozen_teacher"):
            "0c323860371b91a4835096d3987487367b2b094bf81a22b22812524e462a32aa",
    }
    for (benchmark, asset_id), digest in expected_hashes.items():
        asset = next(asset for asset in manifests[benchmark].assets if asset.id == asset_id)
        if digest is None:
            assert asset.integrity.status == "unknown"
            assert asset.integrity.sha256 is None
        else:
            assert asset.integrity.status == "known"
            assert asset.integrity.sha256 == digest


def test_recorded_model_and_panel_hashes_are_preserved() -> None:
    manifests = load_all_asset_manifests()
    expected_hashes = {
        "maniskill": {
            "base_model": "9877be1633a9bc89834d00ef890e1a4cd34a91c155d710b97c92fbab8a5caa89",
            "frozen_teacher": "8445f7c3e5dcfa6d42623b2ce975c94280630a12513f2543584afaf84733cbdf",
            "normalization_stats": (
                "229cea8fece49762eb8c940e95c0e33b62a3f0473ee9085a9e519cdab6dccdd5"
            ),
            "teacher_normalization_stats": (
                "930b816bf2f142f3e377d088d2252e5238b8d082ad8f0abceedfc6296be58852"
            ),
            "evaluation_panel": "b0e19481f6cbf38013ebd243bdc681f6b9dba7a9e9f8f94d02549bb4714d69d0",
        },
        "calvin_abc_d": {
            "base_model": "1ac4fcd76dfe131f9ce4cec94a750424f9eac0275285ebda7ad05ee9550c3bf8",
            "frozen_teacher": "0c323860371b91a4835096d3987487367b2b094bf81a22b22812524e462a32aa",
            "normalization_stats": (
                "72903e3dad8fc7f4f799e36ee723d0c7f53cb815cc6c55f8ad75f62358f00b41"
            ),
            "official_d_panel": "0aaa7c37dd5976b3f4372b8501e67e692ff565ba48730c0481aa3f92ff71ff60",
        },
        "metaworld_mt50": {
            "base_model": "6e877ae8e2c9c33a2b8d1e3a8a60649ef39b0268da60a3ee71bc5e33cbca7221",
            "frozen_teacher": "e14684cd8e6938ec026eba1d3a11dde757a2b5b503194db85c5b51517fcb11c9",
            "normalization_stats": (
                "ab3e2f7380ccbd0967fe391e07d4bc67492ceb7e3e06a9c03397aa6c28309220"
            ),
        },
    }

    for benchmark, assets in expected_hashes.items():
        by_id = {asset.id: asset for asset in manifests[benchmark].assets}
        for asset_id, digest in assets.items():
            assert by_id[asset_id].integrity.status == "known"
            assert by_id[asset_id].integrity.sha256 == digest

    normalization_algorithms = {
        next(
            asset for asset in manifest.assets if asset.id == "normalization_stats"
        ).integrity.algorithm
        for manifest in manifests.values()
    }
    assert normalization_algorithms == {"sha256_file", "sha256_canonical_json"}


def test_all_historical_final_checkpoint_hashes_are_listed() -> None:
    expected = {
        "maniskill": {
            (
                0,
                "endpoint_dagger",
            ): "ce02b5d68760f04d3c9980d93c288cbf4e6194e712509325c7201dfe120cccf6",
            (0, "path_opd"): "f9c08abca320b95215d08ba4a3e9c6a56c89ca9e8bdaf03faa4e9775461eb0c0",
            (
                1,
                "endpoint_dagger",
            ): "62b818b305461432b81ad733333798c51f898b55ba1169adac10b635db9d1aad",
            (1, "path_opd"): "94803b092ddb7d4b0221e51b27d875c817e277eb47932c298b1087017af53a18",
            (
                2,
                "endpoint_dagger",
            ): "9e53cd4ee08c96bd694de4c376d19ab3dab944675b4513cb22995335107389c6",
            (2, "path_opd"): "6b4f0dc54de1d86efc44171f8d6edc541ba6a00e82bea3ada21cee414630d40c",
        },
        "calvin_abc_d": {
            (
                0,
                "endpoint_dagger",
            ): "084939c8b1d95bea590d7397bfc708785af1e49aa229df6190d4acade6fc6b4d",
            (0, "path_opd"): "cb936ad112c6015f020abdccf975db112491b2cd9a4003b6b3bf2e2f0e01c5f3",
            (
                1,
                "endpoint_dagger",
            ): "9d4f471a38fbac6dd6f0427a097bb79412eaf6828860c984f6283c9c91bfcc60",
            (1, "path_opd"): "9909a7007334557d2324425734d20e757ebf828f02fca4f6e876b1343049a7bd",
            (
                2,
                "endpoint_dagger",
            ): "0539b3a35fa0cf03eaab0bf9672d6882c36267a1c8aa9259e1441053f8dc4fb8",
            (2, "path_opd"): "b8ae3e4eb1c67793f4ae9cf6ddda8b305223b337a054d2b1486511be7687e22b",
        },
        "metaworld_mt50": {
            (
                0,
                "endpoint_dagger",
            ): "2af9fbf4052d70a0b4754662cf55df4d600aafbdd3630be0f76f3dcb6dcddc51",
            (0, "path_opd"): "9625f3ee138c56dc94057a35448345de16b6dabb62934f5282f6c7f69a7aaf3e",
            (
                1,
                "endpoint_dagger",
            ): "6a480ceefaff0ac4545bdd3eb47881b18dee105f4ed80953f7df28a3a5e56243",
            (1, "path_opd"): "e5e93898123b965efae2cf1fe0b6223c923a7008f715aaac9073d8983aac68a3",
        },
    }

    for benchmark, expected_checkpoints in expected.items():
        manifest = load_asset_manifest(benchmark)
        actual = {
            (asset.selector.training_seed, asset.selector.method): asset.integrity.sha256
            for asset in manifest.assets
            if asset.kind == "paper_checkpoint" and asset.selector is not None
        }
        assert actual == expected_checkpoints


def test_asset_validator_rejects_repository_path_traversal() -> None:
    payload = _asset_payload()
    assets = payload["assets"]
    assert isinstance(assets, list)
    asset = assets[0]
    assert isinstance(asset, dict)
    location = asset["location"]
    assert isinstance(location, dict)
    location.update(
        {
            "type": "repository_relative",
            "parameter": None,
            "relative_path": "../private/model.bin",
        }
    )

    with pytest.raises(ConfigError, match="relative_path"):
        validate_asset_payload(payload)


def test_asset_resolution_accepts_only_explicit_user_values(tmp_path: Path) -> None:
    manifest = load_asset_manifest("maniskill")
    supplied = {}
    for asset in manifest.assets:
        assert asset.location.parameter is not None
        path = tmp_path / asset.id
        path.write_bytes(b"placeholder")
        supplied[asset.location.parameter] = path

    resolved = resolve_asset_paths(manifest, supplied=supplied)

    assert set(resolved) == {asset.id for asset in manifest.assets}
    assert all(path.is_absolute() for path in resolved.values())


def test_asset_resolution_reports_missing_parameters() -> None:
    manifest = load_asset_manifest("metaworld_mt50")

    with pytest.raises(AssetResolutionError, match="missing required asset parameters"):
        resolve_asset_paths(manifest, supplied={})


def test_integrity_verification_detects_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"released bytes")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    payload = _asset_payload()
    assets = payload["assets"]
    assert isinstance(assets, list)
    asset_payload = assets[0]
    assert isinstance(asset_payload, dict)
    integrity = asset_payload["integrity"]
    assert isinstance(integrity, dict)
    integrity["sha256"] = digest
    asset = validate_asset_payload(payload).assets[0]

    assert verify_asset_integrity(asset, artifact) is True

    artifact.write_bytes(b"modified bytes")
    with pytest.raises(AssetIntegrityError, match="SHA-256 mismatch"):
        verify_asset_integrity(asset, artifact)


def test_canonical_json_integrity_ignores_json_formatting(tmp_path: Path) -> None:
    artifact = tmp_path / "stats.json"
    artifact.write_text('{"b": [2, 3], "a": 1}\n', encoding="utf-8")
    digest = hashlib.sha256(b'{"a":1,"b":[2,3]}').hexdigest()
    payload = _asset_payload()
    assets = payload["assets"]
    assert isinstance(assets, list)
    asset_payload = assets[0]
    assert isinstance(asset_payload, dict)
    integrity = asset_payload["integrity"]
    assert isinstance(integrity, dict)
    integrity.update(
        {
            "algorithm": "sha256_canonical_json",
            "sha256": digest,
        }
    )
    asset = validate_asset_payload(payload).assets[0]

    assert verify_asset_integrity(asset, artifact) is True

    artifact.write_text('{"a": 2, "b": [2, 3]}\n', encoding="utf-8")
    with pytest.raises(AssetIntegrityError, match="SHA-256 mismatch"):
        verify_asset_integrity(asset, artifact)

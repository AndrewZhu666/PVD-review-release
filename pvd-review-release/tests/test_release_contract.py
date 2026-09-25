from __future__ import annotations

import json
from pathlib import Path

import pytest

from path_opd.release_contract import (
    CONTRACT_SCHEMA,
    PINNED_RLINF_REVISION,
    ReleaseContractError,
    build_checkpoint_manifest,
    formal_panel_contract,
    load_provenance_manifest,
    tree_sha256,
    validate_checkpoint_manifest,
    validate_rlinf_host_support,
    verify_panel_digest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_public_provenance_manifest_is_complete_and_loadable() -> None:
    entries = load_provenance_manifest(REPOSITORY_ROOT / "configs" / "provenance.json")
    assert entries
    assert {entry.name for entry in entries} >= {"RLinf", "RLinf-openpi", "PyTorch"}
    rlinf = next(entry for entry in entries if entry.name == "RLinf")
    assert rlinf.url == "https://github.com/RLinf/RLinf"
    assert rlinf.revision == "fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c"
    assert rlinf.license == "Apache-2.0"
    assert len(rlinf.sha256 or "") == 64
    assert rlinf.sha256_scope == "integration_patch"
    maniskill = next(entry for entry in entries if entry.name == "ManiSkill")
    assert maniskill.url == "https://github.com/mani-skill/ManiSkill"
    assert maniskill.revision == "33967b9e3ead1f841eec57cc9f31d0d8b8cf0907"
    assert maniskill.license == "Apache-2.0"


def test_formal_panel_contract_uses_manifest_digest() -> None:
    contract = formal_panel_contract("maniskill")
    assert contract.asset_id == "evaluation_panel"
    assert contract.expected_sha256 == (
        "b0e19481f6cbf38013ebd243bdc681f6b9dba7a9e9f8f94d02549bb4714d69d0"
    )
    result = verify_panel_digest(
        "maniskill",
        contract.expected_sha256,
        mode="formal",
        supplied_sha256=contract.expected_sha256,
        rows=320,
    )
    assert result.formal is True
    assert result.mode == "formal"


def test_formal_panel_rejects_a_custom_digest_and_custom_is_explicit() -> None:
    custom = "a" * 64
    with pytest.raises(ReleaseContractError, match="not the formal release panel"):
        verify_panel_digest("maniskill", custom, mode="formal", supplied_sha256=custom)
    result = verify_panel_digest("maniskill", custom, mode="custom", supplied_sha256=custom)
    assert result.formal is False
    assert result.as_dict()["custom"] is True


def test_meta_world_requires_explicit_custom_panel_until_public_artifact_exists() -> None:
    with pytest.raises(ReleaseContractError, match="no published formal panel digest"):
        verify_panel_digest("metaworld_mt50", "b" * 64, mode="formal")


def test_checkpoint_manifest_binds_assets_and_code() -> None:
    assets = {"base_model": "1" * 64, "normalization_stats": "2" * 64}
    code = {"release_tree": "3" * 64, "rlinf_revision": "fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c"}
    manifest = build_checkpoint_manifest(
        benchmark="maniskill",
        method="path_opd",
        seed=1,
        world_size=8,
        update=4,
        contract={"solver_steps": 8, "execution_prefix": 5},
        assets=assets,
        code=code,
    )
    assert manifest["schema"] == "path-opd-checkpoint-v1"
    validate_checkpoint_manifest(
        manifest,
        expected={"benchmark": "maniskill", "method": "path_opd", "seed": 1, "world_size": 8},
        assets=assets,
        code=code,
    )
    with pytest.raises(ReleaseContractError, match="asset base_model mismatch"):
        validate_checkpoint_manifest(
            manifest,
            expected={"benchmark": "maniskill", "method": "path_opd", "seed": 1, "world_size": 8},
            assets={"base_model": "4" * 64, "normalization_stats": "2" * 64},
            code=code,
        )


def test_tree_digest_is_independent_of_absolute_root(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "nested").mkdir(parents=True)
        (root / "nested" / "file.txt").write_text("same\n", encoding="utf-8")
    assert tree_sha256(left) == tree_sha256(right)


def test_provenance_schema_rejects_non_https_url(tmp_path: Path) -> None:
    payload = {
        "schema": CONTRACT_SCHEMA,
        "sources": [
            {
                "name": "bad",
                "url": "http://example.invalid",
                "revision": None,
                "license": "MIT",
                "sha256": None,
                "redistribution": "external-download",
            }
        ],
    }
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReleaseContractError, match="HTTPS"):
        load_provenance_manifest(path)


def _fake_rlinf_checkout(tmp_path: Path, *, benchmark: str = "maniskill") -> Path:
    checkout = tmp_path / "rlinf"
    files = {
        "rlinf/models/embodiment/action_contract.py": "class ValidActionContract: pass\n",
        "rlinf/models/embodiment/openpi/openpi_action_model.py": (
            "valid_action_contract\ninitial_noise\n"
        ),
    }
    if benchmark == "calvin_abc_d":
        files.update(
            {
                "rlinf/envs/calvin/calvin_gym_env.py": "capture_checkpoint_state\n",
                "rlinf/envs/calvin/venv.py": "restore_from_storage\n",
            }
        )
    for relative, content in files.items():
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return checkout


def test_rlinf_host_support_requires_pinned_revision_and_patch_interfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_rlinf_checkout(tmp_path, benchmark="calvin_abc_d")
    monkeypatch.setattr(
        "path_opd.release_contract.git_revision", lambda _path: PINNED_RLINF_REVISION
    )
    report = validate_rlinf_host_support(checkout, benchmark="calvin_abc_d")
    assert report["observed_revision"] == PINNED_RLINF_REVISION
    assert all(report["host_patch_interfaces"].values())


def test_rlinf_host_support_rejects_unpinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_rlinf_checkout(tmp_path)
    monkeypatch.setattr("path_opd.release_contract.git_revision", lambda _path: "a" * 40)
    with pytest.raises(ReleaseContractError, match="pinned release baseline"):
        validate_rlinf_host_support(checkout, benchmark="maniskill")


def test_rlinf_host_support_rejects_missing_calvin_patch_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_rlinf_checkout(tmp_path, benchmark="maniskill")
    monkeypatch.setattr(
        "path_opd.release_contract.git_revision", lambda _path: PINNED_RLINF_REVISION
    )
    with pytest.raises(ReleaseContractError, match="missing the release host patch"):
        validate_rlinf_host_support(checkout, benchmark="calvin_abc_d")


def test_rlinf_host_support_binds_patch_digest_and_openpi_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_rlinf_checkout(tmp_path)
    monkeypatch.setattr(
        "path_opd.release_contract.git_revision", lambda _path: PINNED_RLINF_REVISION
    )
    patch = REPOSITORY_ROOT / "integrations" / "rlinf" / "path-opd-host-support.patch"
    monkeypatch.setattr(
        "path_opd.release_contract.importlib.metadata.version", lambda _name: "0.1.1"
    )
    monkeypatch.setattr(
        "path_opd.release_contract._exact_patch_is_applied", lambda _checkout, _patch: True
    )
    report = validate_rlinf_host_support(
        checkout,
        benchmark="maniskill",
        patch_path=patch,
        require_openpi_distribution=True,
    )
    assert report["host_patch_file_sha256"] == report["host_patch_sha256"]
    assert report["exact_patch_application"] is True
    assert report["rlinf_openpi_version"] == "0.1.1"

    bad_patch = tmp_path / "host.patch"
    bad_patch.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ReleaseContractError, match="patch SHA-256"):
        validate_rlinf_host_support(checkout, benchmark="maniskill", patch_path=bad_patch)


def test_rlinf_host_support_rejects_partially_applied_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_rlinf_checkout(tmp_path)
    monkeypatch.setattr(
        "path_opd.release_contract.git_revision", lambda _path: PINNED_RLINF_REVISION
    )
    monkeypatch.setattr(
        "path_opd.release_contract._exact_patch_is_applied", lambda _checkout, _patch: False
    )
    patch = REPOSITORY_ROOT / "integrations" / "rlinf" / "path-opd-host-support.patch"

    with pytest.raises(ReleaseContractError, match="complete release host patch"):
        validate_rlinf_host_support(checkout, benchmark="maniskill", patch_path=patch)


def test_rlinf_host_support_rejects_wrong_openpi_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_rlinf_checkout(tmp_path)
    monkeypatch.setattr(
        "path_opd.release_contract.git_revision", lambda _path: PINNED_RLINF_REVISION
    )
    monkeypatch.setattr(
        "path_opd.release_contract.importlib.metadata.version", lambda _name: "9.9.9"
    )
    with pytest.raises(ReleaseContractError, match="rlinf-openpi version"):
        validate_rlinf_host_support(
            checkout,
            benchmark="maniskill",
            require_openpi_distribution=True,
        )

from __future__ import annotations

import importlib.util
import tarfile
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


_ARCHIVE_MODULE_SPEC = importlib.util.spec_from_file_location(
    "build_release_archive",
    REPOSITORY_ROOT / "scripts" / "build_release_archive.py",
)
assert _ARCHIVE_MODULE_SPEC is not None and _ARCHIVE_MODULE_SPEC.loader is not None
_ARCHIVE_MODULE = importlib.util.module_from_spec(_ARCHIVE_MODULE_SPEC)
_ARCHIVE_MODULE_SPEC.loader.exec_module(_ARCHIVE_MODULE)


def test_docker_test_image_contains_integration_evidence() -> None:
    dockerfile = (REPOSITORY_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (REPOSITORY_ROOT / "docker" / "Dockerfile.dockerignore").read_text(
        encoding="utf-8"
    )

    assert "COPY integrations ./integrations" in dockerfile
    assert "COPY third_party ./third_party" in dockerfile
    assert "COPY benchmarks ./benchmarks" in dockerfile
    assert "!integrations/rlinf/**" in dockerignore
    assert "!third_party/rlinf/**" in dockerignore
    assert "!benchmarks/**" in dockerignore


def test_container_entrypoint_exposes_benchmark_inspection() -> None:
    entrypoint = (REPOSITORY_ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")

    assert "benchmark)" in entrypoint
    assert "python -m path_opd.cli benchmark" in entrypoint


def test_packaged_configs_are_exact_mirrors() -> None:
    public_configs = REPOSITORY_ROOT / "configs"
    packaged_configs = REPOSITORY_ROOT / "src" / "path_opd" / "data" / "configs"

    expected = {
        Path("maniskill.json"),
        Path("calvin_abc_d.json"),
        Path("metaworld_mt50.json"),
        Path("assets/maniskill.json"),
        Path("assets/calvin_abc_d.json"),
        Path("assets/metaworld_mt50.json"),
        Path("provenance.json"),
    }
    assert {path.relative_to(public_configs) for path in public_configs.rglob("*.json")} == expected
    assert {
        path.relative_to(packaged_configs) for path in packaged_configs.rglob("*.json")
    } == expected
    for relative_path in expected:
        assert (public_configs / relative_path).read_bytes() == (
            packaged_configs / relative_path
        ).read_bytes()


def test_pyproject_includes_config_package_data() -> None:
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.setuptools.package-data]" in pyproject
    assert '"data/configs/*.json"' in pyproject
    assert '"data/configs/assets/*.json"' in pyproject


def test_release_archive_builder_rejects_generated_and_binary_members(tmp_path: Path) -> None:
    generated = tmp_path / "artifacts"
    generated.mkdir()
    with pytest.raises(_ARCHIVE_MODULE.ArchiveError, match="generated/private"):
        _ARCHIVE_MODULE._validate_member(
            tmp_path, "artifacts/report.json", max_member_bytes=1024
        )

    binary = tmp_path / "model.pt"
    binary.write_bytes(b"checkpoint")
    with pytest.raises(_ARCHIVE_MODULE.ArchiveError, match="binary/archive"):
        _ARCHIVE_MODULE._validate_member(tmp_path, "model.pt", max_member_bytes=1024)

    target = tmp_path / "README.md"
    target.write_text("public\n", encoding="utf-8")
    link = tmp_path / "link.md"
    link.symlink_to(target)
    with pytest.raises(_ARCHIVE_MODULE.ArchiveError, match="symbolic links"):
        _ARCHIVE_MODULE._validate_member(tmp_path, "link.md", max_member_bytes=1024)


def test_release_archive_builder_rejects_anonymity_findings(tmp_path: Path) -> None:
    audit_script = tmp_path / "scripts" / "audit_anonymity.py"
    audit_script.parent.mkdir()
    audit_script.write_bytes((REPOSITORY_ROOT / "scripts" / "audit_anonymity.py").read_bytes())
    (tmp_path / "README.md").write_text(
        # Keep the fixture path out of this repository's own anonymity scan;
        # the temporary tree still receives the complete path at runtime.
        "absolute path: " + "/" + "home/researcher/private-checkpoint\n",
        encoding="utf-8",
    )

    with pytest.raises(_ARCHIVE_MODULE.ArchiveError, match="anonymity audit failed") as error:
        _ARCHIVE_MODULE._run_anonymity_audit(tmp_path)
    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize("relative", ["texture.png", "mesh.obj", "lib/libmujoco.so", "scene.mjcf"])
def test_release_archive_builder_rejects_small_simulator_or_binary_assets(
    tmp_path: Path, relative: str
) -> None:
    candidate = tmp_path / relative
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"small asset")

    with pytest.raises(_ARCHIVE_MODULE.ArchiveError, match="binary/archive"):
        _ARCHIVE_MODULE._validate_member(tmp_path, relative, max_member_bytes=1024)


@pytest.mark.parametrize(
    "relative", [".env", ".env.local", "credentials.json", "keys/id_rsa", "keys/model.pem"]
)
def test_release_archive_builder_rejects_credential_members(
    tmp_path: Path, relative: str
) -> None:
    candidate = tmp_path / relative
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("private\n", encoding="utf-8")

    with pytest.raises(
        _ARCHIVE_MODULE.ArchiveError, match=r"(credential/private|binary/archive)"
    ):
        _ARCHIVE_MODULE._validate_member(tmp_path, relative, max_member_bytes=1024)


def test_release_archive_builder_allows_anonymous_environment_template(tmp_path: Path) -> None:
    template = tmp_path / ".env.example"
    template.write_text("PUBLIC_SETTING=replace-me\n", encoding="utf-8")

    assert _ARCHIVE_MODULE._validate_member(
        tmp_path, ".env.example", max_member_bytes=1024
    ) == template


@pytest.mark.skipif(
    not (REPOSITORY_ROOT / ".git").exists(), reason="archive extraction has no Git metadata"
)
def test_release_archive_builder_normalizes_metadata(tmp_path: Path) -> None:
    archive_path = tmp_path / "release.tar"
    members = _ARCHIVE_MODULE.build_archive(REPOSITORY_ROOT, archive_path)

    assert "scripts/build_release_archive.py" in members
    assert all("artifacts" not in member for member in members)
    with tarfile.open(archive_path) as archive:
        entries = archive.getmembers()
        assert len(entries) == len(members)
        assert all(entry.uid == 0 and entry.gid == 0 for entry in entries)
        assert all(entry.uname == "" and entry.gname == "" for entry in entries)
        assert all(entry.mtime == 0 for entry in entries)
        assert all(entry.isfile() for entry in entries)

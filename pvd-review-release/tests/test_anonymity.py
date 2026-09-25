from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPOSITORY_ROOT / "scripts" / "audit_anonymity.py"


def _load_audit_module():
    specification = importlib.util.spec_from_file_location("audit_anonymity", AUDIT_SCRIPT)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


audit = _load_audit_module()


def _rules(findings) -> set[str]:
    return {finding.rule for finding in findings}


def test_clean_tree_passes(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "Anonymous research artifact with relative paths only.\n", encoding="utf-8"
    )

    assert audit.scan_repository(tmp_path, scan_git=False) == []


def test_private_infrastructure_and_identity_are_detected(tmp_path: Path) -> None:
    tracker_name = "wan" + "db"
    host_name = "no" + "de" + "77"
    experiment_name = "prom" + "pt" + "044"
    user_path = "/" + "home" + "/researcher/private.py"
    email = "researcher" + "@example.org"
    (tmp_path / "config.txt").write_text(
        f"{tracker_name}\n{host_name}\n{experiment_name}\n{user_path}\n{email}",
        encoding="utf-8",
    )

    assert {
        "cluster-hostname",
        "email-address",
        "experiment-identifier",
        "experiment-tracking",
        "unix-user-path",
    }.issubset(_rules(audit.scan_repository(tmp_path, scan_git=False)))


def test_sensitive_file_name_is_scanned(tmp_path: Path) -> None:
    sensitive_name = "no" + "de" + "19" + "_notes.txt"
    (tmp_path / sensitive_name).write_text("neutral content\n", encoding="utf-8")

    findings = audit.scan_repository(tmp_path, scan_git=False)

    assert any(finding.rule == "cluster-hostname" for finding in findings)


def test_third_party_attribution_email_is_preserved(tmp_path: Path) -> None:
    license_directory = tmp_path / "third_party" / "example"
    license_directory.mkdir(parents=True)
    upstream_email = "upstream" + "@example.org"
    (license_directory / "LICENSE").write_text(
        f"Copyright Upstream Project <{upstream_email}>\n", encoding="utf-8"
    )
    (license_directory / "AUTHORS").write_text(
        f"Author: Upstream Maintainer <{upstream_email}>\n", encoding="utf-8"
    )

    assert audit.scan_repository(tmp_path, scan_git=False) == []


def test_project_author_metadata_is_not_exempt_by_filename(tmp_path: Path) -> None:
    (tmp_path / "AUTHORS").write_text("Author: Project Maintainer\n", encoding="utf-8")

    assert "identity-metadata" in _rules(audit.scan_repository(tmp_path, scan_git=False))


def test_email_in_third_party_source_is_not_exempt(tmp_path: Path) -> None:
    source_directory = tmp_path / "third_party" / "example"
    source_directory.mkdir(parents=True)
    upstream_email = "upstream" + "@example.org"
    (source_directory / "module.py").write_text(f'CONTACT = "{upstream_email}"\n', encoding="utf-8")

    assert "email-address" in _rules(audit.scan_repository(tmp_path, scan_git=False))


def test_absolute_symlink_is_detected(tmp_path: Path) -> None:
    target = "/" + "tmp" + "/external-artifact"
    os.symlink(target, tmp_path / "artifact")

    findings = audit.scan_repository(tmp_path, scan_git=False)

    assert any(finding.rule == "absolute-symlink" for finding in findings)


def test_cli_json_output_and_exit_code(tmp_path: Path) -> None:
    private_path = "/" + "data" + "/users/researcher/checkpoint"
    (tmp_path / "manifest.txt").write_text(private_path, encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            str(tmp_path),
            "--format",
            "json",
            "--no-git",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert any(item["rule"] == "shared-user-path" for item in payload)
    assert all("researcher" not in item["context"] for item in payload)


def test_non_anonymous_metadata_is_detected(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        'authors = [{name = "A. Researcher"}]\n', encoding="utf-8"
    )

    assert "identity-metadata" in _rules(audit.scan_repository(tmp_path, scan_git=False))


def test_anonymous_metadata_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        'authors = [{name = "Anonymous Authors"}]\n', encoding="utf-8"
    )

    assert audit.scan_repository(tmp_path, scan_git=False) == []


def test_tracked_file_under_ignored_directory_is_scanned(tmp_path: Path, monkeypatch) -> None:
    # The normal walk skips local virtualenvs/caches for performance.  A file
    # that is already tracked must not inherit that exemption in a release audit.
    nested = tmp_path / ".venv" / "leak.txt"
    nested.parent.mkdir()
    private_path = "/" + "home" + "/researcher/private-checkpoint"
    nested.write_text(f"{private_path}\n", encoding="utf-8")
    monkeypatch.setattr(audit, "_git_tracked_paths", lambda _root: {Path(".venv/leak.txt")})

    findings = audit.scan_repository(tmp_path, scan_git=False)

    assert any(
        finding.path == ".venv/leak.txt" and finding.rule == "unix-user-path"
        for finding in findings
    )

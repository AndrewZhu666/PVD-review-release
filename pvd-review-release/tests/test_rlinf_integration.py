from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PATCH = REPOSITORY_ROOT / "integrations" / "rlinf" / "path-opd-host-support.patch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rlinf_patch_is_frozen_and_targets_only_reviewed_files() -> None:
    assert _sha256(PATCH) == "30296071647c0fcdd36d7b12c08fd5d627b5c2f03e8ac73360def51784b30d6f"
    text = PATCH.read_text(encoding="utf-8")
    targets = set(re.findall(r"^diff --git a/(\S+) b/(\S+)$", text, re.MULTILINE))
    assert targets == {
        (
            "rlinf/envs/calvin/calvin_gym_env.py",
            "rlinf/envs/calvin/calvin_gym_env.py",
        ),
        ("rlinf/envs/calvin/venv.py", "rlinf/envs/calvin/venv.py"),
        (
            "rlinf/models/embodiment/action_contract.py",
            "rlinf/models/embodiment/action_contract.py",
        ),
        (
            "rlinf/models/embodiment/openpi/openpi_action_model.py",
            "rlinf/models/embodiment/openpi/openpi_action_model.py",
        ),
    }


def test_rlinf_patch_records_the_exact_public_baseline() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    assert "fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c" in readme
    assert "git apply --check" in readme
    assert "30296071647c0fcdd36d7b12c08fd5d627b5c2f03e8ac73360def51784b30d6f" in readme


def test_rlinf_license_copy_differs_only_by_final_newline() -> None:
    license_text = (REPOSITORY_ROOT / "third_party" / "rlinf" / "LICENSE").read_text(
        encoding="utf-8"
    )
    assert license_text.startswith("                                 Apache License\n")
    assert license_text.rstrip().endswith("limitations under the License.")
    assert _sha256(REPOSITORY_ROOT / "third_party" / "rlinf" / "LICENSE") == (
        "c1b9df1275e769f3dbab000d1e457a2d4b0f28eb5da6c77e48dc37eeba202ed7"
    )

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar

import pytest

from benchmarks.calvin_abc_d import common as calvin_common
from benchmarks.calvin_abc_d import export_panel as calvin_export
from benchmarks.maniskill import common as maniskill_common
from benchmarks.maniskill import export_panel as maniskill_export


class _FakeManiSkillEnv:
    created: ClassVar[list[_FakeManiSkillEnv]] = []

    def __init__(
        self, *, cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
    ):
        assert num_envs == 40
        assert total_num_processes == 8
        assert worker_info is None
        assert record_metrics is False
        self.reset_state_ids = [seed_offset * 10_000 + slot for slot in range(num_envs)]
        self.reset_calls = 0
        self.closed = False
        self.cfg = cfg
        type(self).created.append(self)

    def reset(self):
        self.reset_calls += 1
        return {}, {}

    def close(self):
        self.closed = True


def _mani_config(rows: int) -> dict[str, int]:
    return {"num_envs": rows}


def test_maniskill_export_uses_real_reset_ids_and_writes_validator_panel(tmp_path: Path) -> None:
    _FakeManiSkillEnv.created.clear()
    output = tmp_path / "maniskill-panel.json"

    report = maniskill_export.export_panel(
        rlinf_checkout=tmp_path / "unused-rlinf",
        simulator_assets=tmp_path / "sim-assets",
        maniskill_package_assets=tmp_path / "ms-assets",
        output=output,
        env_factory=_FakeManiSkillEnv,
        config_factory=_mani_config,
    )

    assert report["row_count"] == 320
    assert report["panel_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert len(_FakeManiSkillEnv.created) == 8
    assert all(item.reset_calls == 1 and item.closed for item in _FakeManiSkillEnv.created)
    panel = maniskill_common.validate_panel(output)
    assert panel["row_count"] == 320
    assert panel["rows"][0] == {
        "env_slot": 0,
        "initial_state_id": 0,
        "pipeline_stage": 0,
        "reset_episode_id": 0,
        "worker_rank": 0,
    }
    assert panel["rows"][-1]["reset_episode_id"] == 70_039


def test_maniskill_export_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    kwargs = {
        "rlinf_checkout": tmp_path / "unused-rlinf",
        "simulator_assets": tmp_path / "sim-assets",
        "maniskill_package_assets": tmp_path / "ms-assets",
        "env_factory": _FakeManiSkillEnv,
        "config_factory": _mani_config,
    }
    maniskill_export.export_panel(output=first, **kwargs)
    maniskill_export.export_panel(output=second, **kwargs)
    assert first.read_bytes() == second.read_bytes()
    assert first.with_name("one.rows.jsonl").read_bytes() == second.with_name(
        "two.rows.jsonl"
    ).read_bytes()


def test_maniskill_export_rejects_duplicate_runtime_ids(tmp_path: Path) -> None:
    class DuplicateEnv(_FakeManiSkillEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.reset_state_ids[0] = self.reset_state_ids[1]

    with pytest.raises(maniskill_common.ContractError, match="duplicate"):
        maniskill_export.collect_reset_rows(
            tmp_path / "unused-rlinf",
            tmp_path / "sim-assets",
            tmp_path / "ms-assets",
            env_factory=DuplicateEnv,
            config_factory=_mani_config,
        )


def _fake_calvin_sequences(count: int):
    return [
        (
            {
                "led": index % 2,
                "lightbulb": 1,
                "slider": "left",
                "drawer": "open",
                "red_block": "table",
                "blue_block": "slider_right",
                "pink_block": "slider_left",
                "grasped": 0,
            },
            [f"task_{index}_{step}" for step in range(5)],
        )
        for index in range(count)
    ]


def test_calvin_export_calls_get_sequences_1000_and_validates_rows(tmp_path: Path) -> None:
    calls: list[int] = []

    def factory(count: int):
        calls.append(count)
        return _fake_calvin_sequences(count)

    output = tmp_path / "calvin-panel.json"
    report = calvin_export.export_panel(
        rlinf_checkout=tmp_path / "unused-rlinf",
        output=output,
        sequence_factory=factory,
    )

    assert calls == [1000]
    assert report["row_count"] == 1000
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rows"][0]["sequence_index"] == 0
    assert payload["rows"][-1]["sequence_index"] == 999
    assert len(payload["rows"][0]["identity_sha256"]) == 64
    panel = calvin_common.validate_panel(output, allow_custom=True)
    assert panel["row_count"] == 1000


def test_calvin_export_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"

    def factory(count: int):
        return _fake_calvin_sequences(count)

    calvin_export.export_panel(
        rlinf_checkout=tmp_path / "unused-rlinf", output=first, sequence_factory=factory
    )
    calvin_export.export_panel(
        rlinf_checkout=tmp_path / "unused-rlinf", output=second, sequence_factory=factory
    )
    assert first.read_bytes() == second.read_bytes()
    assert first.with_name("one.rows.jsonl").read_bytes() == second.with_name(
        "two.rows.jsonl"
    ).read_bytes()


def test_calvin_export_rejects_short_or_duplicate_upstream_output(tmp_path: Path) -> None:
    with pytest.raises(calvin_common.ContractError, match="returned 999"):
        calvin_export.collect_rows(
            tmp_path / "unused-rlinf", sequence_factory=lambda _count: _fake_calvin_sequences(999)
        )

    duplicate = _fake_calvin_sequences(1000)
    duplicate[1] = duplicate[0]
    with pytest.raises(calvin_common.ContractError, match="duplicates"):
        calvin_export.collect_rows(
            tmp_path / "unused-rlinf", sequence_factory=lambda _count: duplicate
        )


def test_exporters_report_missing_runtime_dependencies_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing_maniskill(*_args, **_kwargs):
        raise maniskill_common.ContractError("RLinf checkout is missing")

    monkeypatch.setattr(maniskill_export, "validate_rlinf_checkout", missing_maniskill)
    maniskill_result = maniskill_export.main(
        [
            "--rlinf-checkout",
            str(tmp_path / "missing"),
            "--simulator-assets",
            str(tmp_path / "sim"),
            "--maniskill-package-assets",
            str(tmp_path / "ms"),
            "--output",
            str(tmp_path / "panel.json"),
        ]
    )
    assert maniskill_result == 2
    assert "RLinf checkout is missing" in capsys.readouterr().err

    def missing_calvin(_checkout: Path):
        raise RuntimeError("CALVIN dependencies are missing")

    monkeypatch.setattr(calvin_export, "_load_get_sequences", missing_calvin)
    calvin_result = calvin_export.main(
        [
            "--rlinf-checkout",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "calvin-panel.json"),
        ]
    )
    assert calvin_result == 2
    assert "CALVIN dependencies are missing" in capsys.readouterr().err

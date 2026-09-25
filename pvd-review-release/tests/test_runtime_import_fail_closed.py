from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.calvin_abc_d import evaluate as calvin_evaluate
from benchmarks.calvin_abc_d import train as calvin_train
from benchmarks.metaworld_mt50 import evaluate as metaworld_evaluate
from benchmarks.metaworld_mt50 import train as metaworld_train

_CASES: tuple[tuple[ModuleType, list[str], str, bool], ...] = (
    (
        calvin_train,
        [
            "--method",
            "path_opd",
            "--rlinf-checkout",
            "checkout",
            "--base-model",
            "base",
            "--teacher-model",
            "teacher",
            "--normalization-stats",
            "norm",
            "--environment-assets",
            "assets",
            "--task-oracle-annotations",
            "annotations",
            "--output",
            "output",
        ],
        "calvin-train",
        False,
    ),
    (
        calvin_evaluate,
        [
            "--rlinf-checkout",
            "checkout",
            "--base-model",
            "base",
            "--checkpoint",
            "checkpoint",
            "--normalization-stats",
            "norm",
            "--environment-assets",
            "assets",
            "--task-oracle-annotations",
            "annotations",
            "--panel",
            "panel",
            "--output",
            "output",
            "--method",
            "path_opd",
            "--training-seed",
            "0",
        ],
        "calvin-evaluate",
        True,
    ),
    (
        metaworld_train,
        [
            "--method",
            "path_opd",
            "--rlinf-checkout",
            "checkout",
            "--base",
            "base",
            "--teacher",
            "teacher",
            "--norm",
            "norm",
            "--output",
            "output",
        ],
        "metaworld-train",
        False,
    ),
    (
        metaworld_evaluate,
        [
            "--rlinf-checkout",
            "checkout",
            "--checkpoint",
            "checkpoint",
            "--norm",
            "norm",
            "--output",
            "output",
            "--method",
            "path_opd",
            "--training-seed",
            "0",
        ],
        "metaworld-evaluate",
        False,
    ),
)


@pytest.mark.parametrize("module, argv, label, emits_json", _CASES)
def test_real_entrypoint_converts_missing_runtime_import_to_exit_two(
    module: ModuleType,
    argv: list[str],
    label: str,
    emits_json: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(module, "_validate_args", lambda _args: SimpleNamespace())
    if module is calvin_evaluate:
        monkeypatch.setattr(module, "validate_panel", lambda *_args, **_kwargs: {})

    def raise_import_error(*_args: object) -> int:
        raise ImportError("missing external runtime")

    target = "_run" if hasattr(module, "_run") else "_worker"
    monkeypatch.setattr(module, target, raise_import_error)

    assert module.main(argv) == 2
    error = capsys.readouterr().err
    if emits_json:
        assert '"error_type": "ImportError"' in error
    else:
        assert label in error
        assert "missing external runtime" in error


def test_import_error_regression_helper_is_callable() -> None:
    # Keep the parameter table's callable/module contract obvious to future
    # maintainers without importing any external simulator dependency.
    assert all(callable(module.main) for module, *_ in _CASES)

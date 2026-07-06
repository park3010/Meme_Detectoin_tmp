from __future__ import annotations

import sys
from pathlib import Path

import pytest

from experiments import progress as progress_module
from experiments.progress import ProgressConfig, progress_iter, progress_write


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_disabled_progress_returns_plain_iterable():
    items = [1, 2, 3]
    wrapped = progress_iter(items, desc="disabled", disable=True)
    assert wrapped is items
    assert list(wrapped) == items


def test_enabled_progress_receives_tqdm_options(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []

    class FakeTqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            calls.append(kwargs)

        def __iter__(self):
            return iter(self.iterable)

    monkeypatch.setattr(progress_module, "_load_tqdm", lambda: FakeTqdm)
    cfg = ProgressConfig(disable=False, mininterval=0.25, leave_batch=True, dynamic_ncols=False)

    wrapped = progress_iter([1], desc="visible", config=cfg, total=1, position=2, leave=True)

    assert list(wrapped) == [1]
    assert calls == [
        {
            "desc": "visible",
            "disable": False,
            "leave": True,
            "total": 1,
            "position": 2,
            "mininterval": 0.25,
            "dynamic_ncols": False,
        }
    ]


def test_tqdm_import_failure_degrades_gracefully(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(progress_module, "_load_tqdm", lambda: None)
    wrapped = progress_iter([1, 2], desc="fallback", config=ProgressConfig(disable=False))
    assert list(wrapped) == [1, 2]


def test_progress_write_safe_without_tqdm(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(progress_module, "_load_tqdm", lambda: None)
    progress_write("hello", config=ProgressConfig(disable=False))
    assert "hello" in capsys.readouterr().out


def test_cli_progress_flags_propagate_to_train_config(monkeypatch: pytest.MonkeyPatch):
    import run
    from commands import experiment as experiment_commands

    captured = {}

    def fake_train(config):
        captured["config"] = config
        return {"macro_f1": 1.0, "accuracy": 1.0}

    monkeypatch.setattr(experiment_commands, "run_ours_experiment", fake_train)
    parser = run.build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset",
            "harm_c",
            "--seed",
            "42",
            "--disable-tqdm",
            "--tqdm-mininterval",
            "0.2",
        ]
    )
    args.func(args)

    cfg = captured["config"]
    assert cfg.progress.disable is True
    assert cfg.progress.mininterval == 0.2


def test_cli_progress_flags_propagate_to_suite_args(monkeypatch: pytest.MonkeyPatch):
    import run
    from commands import experiment as experiment_commands

    captured = {}

    def fake_suite(args):
        captured["progress"] = args.progress_config
        return {"status": "complete"}

    monkeypatch.setattr(experiment_commands, "run_suite", fake_suite)
    parser = run.build_parser()
    args = parser.parse_args(["suite", "--suite", "core_smoke", "--tqdm-mininterval", "0.3"])
    args.func(args)

    assert captured["progress"].disable is None
    assert captured["progress"].mininterval == 0.3

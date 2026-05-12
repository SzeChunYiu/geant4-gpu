"""Tests for the V7 validation runner scaffold."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_run_v7():
    module_path = Path(__file__).with_name("run_v7.py")
    spec = importlib.util.spec_from_file_location("run_v7", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_loader_is_explicitly_blocked_until_v7_reader_exists(tmp_path):
    run_v7 = _load_run_v7()

    try:
        run_v7.load_observable_table(tmp_path / "candidate.root")
    except NotImplementedError as exc:
        message = str(exc)
    else:
        raise AssertionError("ROOT loading should be an explicit V7 blocker")

    assert "docs/VALIDATION.md V7" in message
    assert "ROOT" in message


def test_evaluate_arrays_reports_tolerance_failure_for_low_ks_pvalue():
    run_v7 = _load_run_v7()

    result = run_v7.evaluate_arrays(
        test="V7",
        observable="tpc.total_edep",
        candidate=[1.0, 2.0, 3.0, 4.0],
        reference=[1.0, 2.0, 3.0, 4.0],
        ks_func=lambda left, right: 0.01,
    )

    assert result["status"] == "FAIL"
    assert result["ks_pvalue"] == 0.01
    assert result["mean_delta"] == 0.0
    assert result["rms_delta"] == 0.0


def test_main_writes_not_implemented_summary(tmp_path):
    run_v7 = _load_run_v7()
    output = tmp_path / "v7_summary.json"

    status = run_v7.main(
        [
            "--candidate",
            str(tmp_path / "candidate.root"),
            "--reference",
            str(tmp_path / "reference.root"),
            "--output",
            str(output),
        ]
    )

    assert status == 2
    payload = json.loads(output.read_text())
    assert payload["test"] == "V7"
    assert payload["status"] == "NOT_IMPLEMENTED"
    assert "docs/VALIDATION.md V7" in payload["error"]

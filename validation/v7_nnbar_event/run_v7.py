#!/usr/bin/env python3
"""Skeleton runner for G4GPU V7 full-event validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

KS_THRESHOLD = 0.05
TEST_NAME = "V7"
VALIDATION_REF = "docs/VALIDATION.md V7"
DEFAULT_OUTPUT = Path("output/v7_summary.json")


def _as_floats(values: Iterable[float]) -> list[float]:
    """Return values as a finite float list."""

    result = [float(value) for value in values]
    if not result:
        raise ValueError("observable arrays must not be empty")
    if any(not math.isfinite(value) for value in result):
        raise ValueError("observable arrays must contain only finite values")
    return result


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean for a non-empty sequence."""

    return sum(values) / len(values)


def _rms(values: Sequence[float]) -> float:
    """Return the population RMS width around the mean."""

    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _default_ks_pvalue(candidate: Sequence[float], reference: Sequence[float]) -> float:
    """Return the two-sample KS p-value using SciPy."""

    try:
        from scipy.stats import ks_2samp
    except ImportError as exc:  # pragma: no cover - depends on LUNARC env
        raise RuntimeError("scipy is required once V7 readers are implemented") from exc
    return float(ks_2samp(candidate, reference).pvalue)


def _mean_tolerance(reference: Sequence[float]) -> float:
    """Return the 1-sigma statistical tolerance for the reference mean."""

    return _rms(reference) / math.sqrt(len(reference)) if len(reference) > 1 else 0.0


def _rms_tolerance(reference: Sequence[float]) -> float:
    """Return the 2-sigma statistical tolerance for the reference RMS."""

    if len(reference) <= 1:
        return 0.0
    return 2.0 * _rms(reference) / math.sqrt(2.0 * (len(reference) - 1))


def evaluate_arrays(
    *,
    test: str,
    observable: str,
    candidate: Iterable[float],
    reference: Iterable[float],
    ks_func: Callable[[Sequence[float], Sequence[float]], float] | None = None,
) -> dict[str, float | str]:
    """Evaluate one scalar V7 observable against Geant4 reference values."""

    candidate_values = _as_floats(candidate)
    reference_values = _as_floats(reference)
    ks_pvalue = (ks_func or _default_ks_pvalue)(candidate_values, reference_values)
    candidate_mean = _mean(candidate_values)
    reference_mean = _mean(reference_values)
    candidate_rms = _rms(candidate_values)
    reference_rms = _rms(reference_values)
    mean_delta = candidate_mean - reference_mean
    rms_delta = candidate_rms - reference_rms
    mean_limit = _mean_tolerance(reference_values)
    rms_limit = _rms_tolerance(reference_values)
    status = "PASS"
    if ks_pvalue <= KS_THRESHOLD:
        status = "FAIL"
    if abs(mean_delta) > mean_limit:
        status = "FAIL"
    if abs(rms_delta) > rms_limit:
        status = "FAIL"
    return {
        "test": test,
        "observable": observable,
        "status": status,
        "ks_pvalue": float(ks_pvalue),
        "mean_delta": float(mean_delta),
        "rms_delta": float(rms_delta),
        "mean_tolerance": float(mean_limit),
        "rms_tolerance": float(rms_limit),
    }


def load_observable_table(path: Path) -> Mapping[str, Sequence[float]]:
    """Load V7 per-sub-detector observables from a ROOT or Parquet output."""

    suffix = path.suffix.lower()
    if suffix == ".root":
        raise NotImplementedError(
            f"ROOT observable loading is deferred to the next V7 wiring "
            f"iteration; see {VALIDATION_REF}."
        )
    if suffix in {".parquet", ".pq"}:
        raise NotImplementedError(
            f"Parquet observable loading is deferred to the next V7 wiring "
            f"iteration; see {VALIDATION_REF}."
        )
    raise ValueError(f"unsupported V7 input format for {path}: expected ROOT or Parquet")


def compare_tables(
    candidate: Mapping[str, Sequence[float]],
    reference: Mapping[str, Sequence[float]],
) -> list[dict[str, float | str]]:
    """Compare matching observable arrays and return JSON-ready records."""

    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference))
    if missing or extra:
        raise ValueError(f"observable key mismatch: missing={missing}, extra={extra}")
    return [
        evaluate_arrays(
            test=TEST_NAME,
            observable=observable,
            candidate=candidate[observable],
            reference=reference[observable],
        )
        for observable in sorted(reference)
    ]


def write_summary(path: Path, payload: object) -> None:
    """Write an indented JSON summary, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the V7 scaffold."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path, help="G4GPU ROOT/Parquet output")
    parser.add_argument("--reference", required=True, type=Path, help="Geant4 reference ROOT/Parquet output")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path, help="JSON summary path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the V7 comparison and return a process status code."""

    args = build_parser().parse_args(argv)
    try:
        candidate = load_observable_table(args.candidate)
        reference = load_observable_table(args.reference)
        records = compare_tables(candidate, reference)
    except NotImplementedError as exc:
        write_summary(
            args.output,
            {"test": TEST_NAME, "status": "NOT_IMPLEMENTED", "error": str(exc)},
        )
        print(str(exc))
        return 2
    except Exception as exc:
        write_summary(args.output, {"test": TEST_NAME, "status": "ERROR", "error": str(exc)})
        print(str(exc))
        return 2
    write_summary(args.output, records)
    return 1 if any(record["status"] != "PASS" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())

# V7 full-event validation scaffold

This directory defines the V7 end-to-end validation harness from
`docs/VALIDATION.md`: compare an opt-in G4GPU full-event output against a
vanilla Geant4 reference for an NNBAR-shaped 10 GeV cosmic-muon sample. The
current compact unit is a scaffold only. It records the observable contract,
output names, acceptance criteria, and failure modes before the real ROOT or
Parquet readers are wired.

## Primary sample

- Primary: 10 GeV cosmic muon entering the detector envelope.
- Statistics: 10,000 G4GPU candidate events and 10,000 vanilla Geant4 reference
  events, both with fixed, recorded seeds.
- Geometry shape: silicon tracker, beam pipe, TPC volume, scintillator system,
  and lead-glass calorimeter are treated as named sub-detectors in the
  validation output. The harness consumes output data only; it does not include,
  link, or call thesis-production detector code.

## Required output paths

- Candidate: `output/v7_g4gpu.root`
- Reference: `output/v7_geant4_ref.root`
- Summary: `output/v7_summary.json`

Parquet inputs may be used in place of ROOT when both candidate and reference
carry the same observable keys.

## V7 observables

Each sub-detector emits the following arrays:

1. `total_edep`: per-event total deposited energy.
2. `hit_multiplicity`: per-event hit count.
3. `hit_xy`: per-layer two-dimensional hit-position distribution in x-y.

The JSON summary records one row per `{sub_detector}.{observable}` comparison.
The skeleton runner currently implements the scalar tolerance bookkeeping for
`total_edep` and `hit_multiplicity`; the real reader iteration must add the
2D x-y statistic for `hit_xy`.

## Acceptance criteria

For every observable in every sub-detector:

- Kolmogorov-Smirnov p-value must be greater than `0.05`.
- Candidate mean must be within one reference statistical sigma.
- Candidate RMS must be within two reference statistical sigmas.

A single tolerance breach makes the runner exit nonzero and marks that
observable `FAIL` in `output/v7_summary.json`.

## Explicit failure modes

- `NOT_IMPLEMENTED`: ROOT/Parquet readers are not wired yet. This is the
  expected status for the first shakedown SLURM job from this scaffold.
- `ERROR`: the candidate/reference files have unsupported formats, mismatched
  observable keys, empty arrays, non-finite values, or missing runtime
  dependencies.
- `FAIL`: inputs load successfully but at least one V7 acceptance tolerance is
  breached.
- `PASS`: every listed observable satisfies the V7 acceptance criteria.

No 10,000-event production validation or B1-B5 benchmark sweep is part of this
iteration. Those runs require a separate planner-approved compact unit after the
readers and event generators are connected.

# B1-B5 benchmark scaffold for V7

The benchmarks below follow `docs/VALIDATION.md` and are intentionally
scaffolded without launching any throughput sweep in this compact unit. Each
benchmark uses the same fixed-release build, recorded GPU model, driver, CUDA
runtime, Geant4 reference build, event seed set, and output manifest.

## B1 — Throughput vs. batch size

- Sweep batch sizes: 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, and 256K tracks or
  track-steps, depending on the component under test.
- Metric: sustained track-steps per second after initialization warm-up.
- Acceptance plot: throughput vs. batch size, with the knee annotated and a
  note when memory bandwidth saturation begins.

## B2 — Speedup vs. Geant4

- Compare G4GPU and vanilla Geant4 on the same LUNARC node class when possible.
- Metric: `speedup = G4GPU track-steps/sec / Geant4 track-steps/sec`.
- Components: muon step, EM shower, optical photon, and full-event V7 once each
  component has a validated implementation.
- Acceptance plot: bar chart with uncertainty bands from repeated fixed-seed
  runs.

## B3 — GPU hardware utilization

- Tooling: NVIDIA Nsight Compute for kernel-level memory bandwidth, SM
  occupancy, and arithmetic intensity.
- Targets: greater than 70% of peak memory bandwidth and greater than 50% SM
  occupancy on the reference A100/A40-class GPU, or an explicit bottleneck note.
- Acceptance plot: utilization panel per kernel with target thresholds drawn.

## B4 — End-to-end event speedup

- Sample: full NNBAR-shaped 10 GeV cosmic-muon V7 events.
- Metric: wall-clock seconds per event, excluding one-time build and staging but
  including runtime data movement needed by the opt-in backend.
- Aggregation: geometric mean speedup across 10,000 events, with a separate
  p50/p95 latency table to expose tail regressions.
- Acceptance plot: event-time ratio panel, G4GPU divided by vanilla Geant4.

## B5 — Scaling with GPU model

- GPU set from `docs/VALIDATION.md`: V100, A100, and RTX 3090 when available;
  record LUNARC substitutions such as A40 if those are the actual test devices.
- Metric: throughput vs. peak memory bandwidth for each GPU.
- Expectation: approximately linear scaling for bandwidth-limited kernels;
  deviations become bottleneck-database entries before publication.
- Acceptance plot: throughput over peak bandwidth, with one point per GPU and
  the fitted scaling slope.

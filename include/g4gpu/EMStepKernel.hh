#pragma once

// GPU EM/gamma kernel interface.
//
// Contract:
// - TrackSOA fields are in millimetres, MeV, nanoseconds, PDG codes, and status
//   values matching G4GPUTrackBuffer.hh.
// - MaterialData points to the active GPU material table used for process
//   selection in later Phase-2 iterations.
// - curandState supplies one RNG state per active track or validation sample.
// - Device samplers return process-level stochastic outcomes; host launchers
//   only manage CUDA grids/RNG setup and do not invoke Geant4 or NNBAR code.

#include "g4gpu/G4GPUCudaCompat.hh"
#include "g4gpu/G4GPUTrackBuffer.hh"
#include "g4gpu/MaterialData.hh"

#if defined(__CUDACC__)
#  include <curand_kernel.h>
#  define G4GPU_DEVICE __device__
#else
struct curandStateXORWOW;
using curandState = curandStateXORWOW;
#  define G4GPU_DEVICE
#endif

namespace g4gpu {

struct ComptonSample {
    float scattered_energy_mev = 0.0f;
    float cos_theta = 1.0f;
    float phi = 0.0f;
};

// Phase 2 EM/gamma per-track step skeleton. Current implementation wires the
// gamma Compton sampler only; process selection, cross-section tables, and
// secondary emission are deferred to SPEC.md Phase 2 follow-up work.
G4GPU_DEVICE void EMStep(
    TrackSOA tracks,
    const MaterialData* materials,
    curandState* rng,
    int n_tracks
);

// SPEC.md Phase 2.N stubs. They document the future device-callable process
// API while intentionally leaving production physics fail-closed.
G4GPU_DEVICE void SamplePhotoelectric(
    TrackSOA tracks,
    const MaterialData* materials,
    curandState* rng,
    int track_index
);
G4GPU_DEVICE ComptonSample SampleCompton(float incident_energy_mev, curandState* rng);
G4GPU_DEVICE void SamplePairProduction(
    TrackSOA tracks,
    const MaterialData* materials,
    curandState* rng,
    int track_index
);
G4GPU_DEVICE void SampleBremsstrahlung(
    TrackSOA tracks,
    const MaterialData* materials,
    curandState* rng,
    int track_index
);

void LaunchEMStepKernel(
    TrackSOA* d_tracks,
    const MaterialData* d_mats,
    curandState* d_rng,
    int n_tracks,
    cudaStream_t stream = nullptr
);

void LaunchComptonSamplingValidationKernel(
    float* d_scattered_energy_mev,
    int n_samples,
    float incident_energy_mev,
    unsigned long long seed,
    cudaStream_t stream = nullptr
);

}  // namespace g4gpu

#undef G4GPU_DEVICE

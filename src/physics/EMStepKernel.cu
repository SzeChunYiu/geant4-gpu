#include "g4gpu/EMStepKernel.hh"

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace g4gpu {
namespace {

constexpr float ELECTRON_MASS_MEV = 0.51099895f;
constexpr float PI = 3.14159265358979323846f;
constexpr int THREADS_PER_BLOCK = 256;

__host__ __device__ float3 Vec(float x, float y, float z) {
    float3 out{x, y, z};
    return out;
}

__host__ __device__ float Dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 Cross(float3 a, float3 b) {
    return Vec(a.y * b.z - a.z * b.y,
               a.z * b.x - a.x * b.z,
               a.x * b.y - a.y * b.x);
}

__host__ __device__ float3 Normalize(float3 v) {
    const float n2 = Dot(v, v);
    if (n2 <= 0.0f) return Vec(0.0f, 0.0f, 1.0f);
    const float inv = rsqrtf(n2);
    return Vec(v.x * inv, v.y * inv, v.z * inv);
}

__device__ void OrthonormalBasis(float3 dir, float3& u, float3& v) {
    const float3 ref = fabsf(dir.z) < 0.9f ? Vec(0.0f, 0.0f, 1.0f)
                                           : Vec(1.0f, 0.0f, 0.0f);
    u = Normalize(Cross(ref, dir));
    v = Normalize(Cross(dir, u));
}

__device__ float SafeUniform(curandState* rng) {
    return rng ? fminf(fmaxf(curand_uniform(rng), 1.0e-7f), 0.99999994f) : 0.5f;
}

__device__ float KleinNishinaPdf(float epsilon, float alpha, float& cos_theta) {
    cos_theta = 1.0f - (1.0f / epsilon - 1.0f) / alpha;
    cos_theta = fminf(1.0f, fmaxf(-1.0f, cos_theta));
    const float sin2 = fmaxf(0.0f, 1.0f - cos_theta * cos_theta);
    return epsilon + 1.0f / epsilon - sin2;
}

__global__ void InitRNGKernel(curandState* rng, int n, unsigned long long seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, i, 0, &rng[i]);
}

__global__ void EMStepKernel(TrackSOA tracks, const MaterialData* materials,
                             curandState* rng, int n_tracks) {
    (void)materials;
    EMStep(tracks, materials, rng, n_tracks);
}

__global__ void ComptonValidationKernel(float* out, int n_samples,
                                        float incident_energy_mev,
                                        curandState* rng) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) return;
    out[i] = SampleCompton(incident_energy_mev, &rng[i]).scattered_energy_mev;
}

}  // namespace

__device__ void SamplePhotoelectric(TrackSOA tracks, const MaterialData* materials,
                                    curandState* rng, int track_index) {
    (void)tracks;
    (void)materials;
    (void)rng;
    (void)track_index;
    // TODO Phase 2.1 (SPEC.md): absorb photon, emit photoelectron/Auger
    // secondaries from tabulated shell cross sections.
    return;
}

__device__ ComptonSample SampleCompton(float incident_energy_mev, curandState* rng) {
    ComptonSample sample{};
    if (incident_energy_mev <= 0.0f) return sample;

    const float alpha = incident_energy_mev / ELECTRON_MASS_MEV;
    const float epsilon_min = 1.0f / (1.0f + 2.0f * alpha);
    const float pdf_bound = epsilon_min + 1.0f / epsilon_min;

    for (int attempt = 0; attempt < 10000; ++attempt) {
        const float epsilon = epsilon_min + (1.0f - epsilon_min) * SafeUniform(rng);
        float cos_theta = 1.0f;
        const float pdf = KleinNishinaPdf(epsilon, alpha, cos_theta);
        if (SafeUniform(rng) * pdf_bound <= pdf) {
            sample.scattered_energy_mev = epsilon * incident_energy_mev;
            sample.cos_theta = cos_theta;
            sample.phi = 2.0f * PI * SafeUniform(rng);
            return sample;
        }
    }

    sample.scattered_energy_mev = incident_energy_mev;
    sample.cos_theta = 1.0f;
    sample.phi = 0.0f;
    return sample;
}

__device__ void SamplePairProduction(TrackSOA tracks, const MaterialData* materials,
                                     curandState* rng, int track_index) {
    (void)tracks;
    (void)materials;
    (void)rng;
    (void)track_index;
    // TODO Phase 2.2 (SPEC.md): gamma -> e+e- threshold/process sampling.
    return;
}

__device__ void SampleBremsstrahlung(TrackSOA tracks, const MaterialData* materials,
                                     curandState* rng, int track_index) {
    (void)tracks;
    (void)materials;
    (void)rng;
    (void)track_index;
    // TODO Phase 2.3 (SPEC.md): e-/e+ bremsstrahlung spectrum and secondaries.
    return;
}

__device__ void EMStep(TrackSOA tracks, const MaterialData* materials,
                       curandState* rng, int n_tracks) {
    (void)materials;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tracks || i >= tracks.size || tracks.status[i] != 0) return;

    const int pdg = tracks.pdg[i];
    if (pdg == 22) {
        curandState* local_rng = rng ? &rng[i] : nullptr;
        const ComptonSample sample = SampleCompton(tracks.ekin[i], local_rng);
        tracks.ekin[i] = sample.scattered_energy_mev;

        float3 dir = Normalize(Vec(tracks.dx[i], tracks.dy[i], tracks.dz[i]));
        float3 u_axis;
        float3 v_axis;
        OrthonormalBasis(dir, u_axis, v_axis);
        const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - sample.cos_theta * sample.cos_theta));
        dir = Normalize(Vec(sample.cos_theta * dir.x +
                                sin_theta * (cosf(sample.phi) * u_axis.x +
                                             sinf(sample.phi) * v_axis.x),
                            sample.cos_theta * dir.y +
                                sin_theta * (cosf(sample.phi) * u_axis.y +
                                             sinf(sample.phi) * v_axis.y),
                            sample.cos_theta * dir.z +
                                sin_theta * (cosf(sample.phi) * u_axis.z +
                                             sinf(sample.phi) * v_axis.z)));
        tracks.dx[i] = dir.x;
        tracks.dy[i] = dir.y;
        tracks.dz[i] = dir.z;
        // TODO Phase 2.4 (SPEC.md): add recoil electron secondary and material
        // cross-section based process selection instead of forced Compton.
        return;
    }

    if (pdg == 11 || pdg == -11) {
        SampleBremsstrahlung(tracks, materials, rng, i);
    }
}

void LaunchEMStepKernel(TrackSOA* d_tracks, const MaterialData* d_mats,
                        curandState* d_rng, int n_tracks, cudaStream_t stream) {
    if (!d_tracks || n_tracks <= 0) return;
    const int blocks = (n_tracks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    EMStepKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(*d_tracks, d_mats, d_rng, n_tracks);
    CheckCuda(cudaGetLastError(), "EMStepKernel launch");
}

void LaunchComptonSamplingValidationKernel(float* d_scattered_energy_mev,
                                           int n_samples,
                                           float incident_energy_mev,
                                           unsigned long long seed,
                                           cudaStream_t stream) {
    if (!d_scattered_energy_mev || n_samples <= 0) return;
    curandState* d_rng = nullptr;
    CheckCuda(cudaMalloc(&d_rng, static_cast<std::size_t>(n_samples) * sizeof(curandState)),
              "cudaMalloc EM rng states");
    const int blocks = (n_samples + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    InitRNGKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_rng, n_samples, seed);
    CheckCuda(cudaGetLastError(), "InitRNGKernel EM launch");
    ComptonValidationKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_scattered_energy_mev, n_samples, incident_energy_mev, d_rng);
    CheckCuda(cudaGetLastError(), "ComptonValidationKernel launch");
    CheckCuda(cudaFree(d_rng), "cudaFree EM rng states");
}

}  // namespace g4gpu

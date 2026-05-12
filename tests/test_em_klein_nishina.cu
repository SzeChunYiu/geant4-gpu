#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include "g4gpu/EMStepKernel.hh"

namespace {

constexpr int kSamples = 10000;
constexpr float kIncidentEnergyMeV = 1.0f;
constexpr double kElectronMassMeV = 0.51099895;
const double kKSCriticalP005 = 1.36 / std::sqrt(static_cast<double>(kSamples));

void Check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << "FAIL: " << what << ": " << cudaGetErrorString(err) << '\n';
        std::exit(1);
    }
}

double KleinNishinaPdf(double scattered_energy_mev) {
    const double alpha = kIncidentEnergyMeV / kElectronMassMeV;
    const double eps = scattered_energy_mev / kIncidentEnergyMeV;
    const double eps_min = 1.0 / (1.0 + 2.0 * alpha);
    if (eps < eps_min || eps > 1.0) return 0.0;
    const double mu = 1.0 - (1.0 / eps - 1.0) / alpha;
    const double sin2 = std::max(0.0, 1.0 - mu * mu);
    return eps + 1.0 / eps - sin2;
}

std::vector<double> BuildReferenceCdf(const std::vector<double>& sorted_samples) {
    constexpr int kGrid = 4096;
    const double alpha = kIncidentEnergyMeV / kElectronMassMeV;
    const double emin = kIncidentEnergyMeV / (1.0 + 2.0 * alpha);
    const double emax = kIncidentEnergyMeV;
    const double h = (emax - emin) / static_cast<double>(kGrid);
    std::vector<double> grid_cdf(kGrid + 1, 0.0);
    for (int i = 1; i <= kGrid; ++i) {
        const double e0 = emin + h * static_cast<double>(i - 1);
        const double e1 = emin + h * static_cast<double>(i);
        grid_cdf[i] = grid_cdf[i - 1] + 0.5 * h * (KleinNishinaPdf(e0) + KleinNishinaPdf(e1));
    }
    const double norm = grid_cdf.back();
    std::vector<double> cdf;
    cdf.reserve(sorted_samples.size());
    for (double e : sorted_samples) {
        if (e <= emin) {
            cdf.push_back(0.0);
            continue;
        }
        if (e >= emax) {
            cdf.push_back(1.0);
            continue;
        }
        const double x = (e - emin) / h;
        const int bin = static_cast<int>(std::floor(x));
        const double frac = x - static_cast<double>(bin);
        const double interpolated = grid_cdf[bin] * (1.0 - frac) + grid_cdf[bin + 1] * frac;
        cdf.push_back(interpolated / norm);
    }
    return cdf;
}

double KolmogorovSmirnovD(const std::vector<float>& samples) {
    std::vector<double> sorted(samples.begin(), samples.end());
    std::sort(sorted.begin(), sorted.end());
    const auto reference_cdf = BuildReferenceCdf(sorted);
    double d = 0.0;
    for (int i = 0; i < kSamples; ++i) {
        const double empirical_hi = static_cast<double>(i + 1) / static_cast<double>(kSamples);
        const double empirical_lo = static_cast<double>(i) / static_cast<double>(kSamples);
        d = std::max(d, std::abs(empirical_hi - reference_cdf[i]));
        d = std::max(d, std::abs(reference_cdf[i] - empirical_lo));
    }
    return d;
}

}  // namespace

int main() {
    float* d_samples = nullptr;
    Check(cudaMalloc(&d_samples, kSamples * sizeof(float)), "cudaMalloc samples");
    g4gpu::LaunchComptonSamplingValidationKernel(
        d_samples, kSamples, kIncidentEnergyMeV, 0xC0FFEEULL, nullptr);

    std::vector<float> samples(kSamples);
    Check(cudaMemcpy(samples.data(), d_samples, kSamples * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy samples");
    Check(cudaFree(d_samples), "cudaFree samples");

    const double d = KolmogorovSmirnovD(samples);
    if (!(d < kKSCriticalP005)) {
        std::cerr << "FAIL: Klein-Nishina KS D=" << d
                  << " critical=" << kKSCriticalP005 << '\n';
        return 1;
    }

    std::cout << "PASS: Klein-Nishina KS D=" << d
              << " critical=" << kKSCriticalP005 << '\n';
    return 0;
}

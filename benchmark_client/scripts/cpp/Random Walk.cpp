#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kWalkers = 1'000'000;
constexpr int kSteps = 1000;
constexpr int kBins2d = 100;

}  // namespace

int main() {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> step_dist(0, 1);

    std::vector<double> sum_pos(kSteps, 0.0);
    std::vector<double> sum_pos_sq(kSteps, 0.0);
    std::vector<long long> hist_1d(2 * kSteps + 1, 0);

    std::vector<double> sum_r2(kSteps, 0.0);
    std::vector<unsigned long long> hist_2d(kBins2d, 0);

    const auto t0 = std::chrono::high_resolution_clock::now();

    for (std::size_t w = 0; w < kWalkers; ++w) {
        int pos = 0;
        for (int s = 0; s < kSteps; ++s) {
            pos += (step_dist(rng) == 0) ? -1 : 1;
            sum_pos[s] += static_cast<double>(pos);
            sum_pos_sq[s] += static_cast<double>(pos) * static_cast<double>(pos);
        }
        hist_1d[static_cast<std::size_t>(pos + kSteps)] += 1;
    }

    const double max_r = std::sqrt(2.0 * kSteps);

    for (std::size_t w = 0; w < kWalkers; ++w) {
        int x = 0;
        int y = 0;
        for (int s = 0; s < kSteps; ++s) {
            x += (step_dist(rng) == 0) ? -1 : 1;
            y += (step_dist(rng) == 0) ? -1 : 1;
            const double r2 = static_cast<double>(x) * static_cast<double>(x) +
                              static_cast<double>(y) * static_cast<double>(y);
            sum_r2[s] += r2;
        }
        const double r = std::sqrt(static_cast<double>(x) * static_cast<double>(x) +
                                   static_cast<double>(y) * static_cast<double>(y));
        int bin = static_cast<int>(r / max_r * kBins2d);
        if (bin < 0) {
            bin = 0;
        }
        if (bin >= kBins2d) {
            bin = kBins2d - 1;
        }
        hist_2d[static_cast<std::size_t>(bin)] += 1ULL;
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("{\"execution_time_ms\": %.6f}\n", elapsed_ms);

    FILE *f_sigma = std::fopen("sigma_1d.csv", "w");
    if (f_sigma) {
        std::fprintf(f_sigma, "step,sigma_empirical,sigma_theory\n");
        for (int s = 0; s < kSteps; ++s) {
            const double mean = sum_pos[s] / static_cast<double>(kWalkers);
            const double mean_sq = sum_pos_sq[s] / static_cast<double>(kWalkers);
            double var = mean_sq - mean * mean;
            if (var < 0 && var > -1e-12) {
                var = 0;
            }
            const double sigma = std::sqrt(var);
            std::fprintf(f_sigma, "%d,%.10g,%.10g\n", s + 1, sigma,
                         std::sqrt(static_cast<double>(s + 1)));
        }
        std::fclose(f_sigma);
    }

    FILE *f_hist = std::fopen("histogram_1d.csv", "w");
    if (f_hist) {
        std::fprintf(f_hist, "position,count\n");
        for (int i = 0; i < 2 * kSteps + 1; ++i) {
            std::fprintf(f_hist, "%d,%lld\n", i - kSteps,
                         static_cast<long long>(hist_1d[static_cast<std::size_t>(i)]));
        }
        std::fclose(f_hist);
    }

    FILE *f_sigma2 = std::fopen("sigma_2d.csv", "w");
    if (f_sigma2) {
        std::fprintf(f_sigma2, "step,sigma_empirical,sigma_theory\n");
        for (int s = 0; s < kSteps; ++s) {
            const double mean_r2 = sum_r2[s] / static_cast<double>(kWalkers);
            std::fprintf(f_sigma2, "%d,%.10g,%.10g\n", s + 1, std::sqrt(mean_r2),
                         std::sqrt(2.0 * static_cast<double>(s + 1)));
        }
        std::fclose(f_sigma2);
    }

    FILE *f_hist2 = std::fopen("histogram_2d.csv", "w");
    if (f_hist2) {
        std::fprintf(f_hist2, "bin,r_min,r_max,count\n");
        for (int b = 0; b < kBins2d; ++b) {
            const double rmin = static_cast<double>(b) * max_r / static_cast<double>(kBins2d);
            const double rmax =
                static_cast<double>(b + 1) * max_r / static_cast<double>(kBins2d);
            std::fprintf(f_hist2, "%d,%.5f,%.5f,%llu\n", b, rmin, rmax,
                         static_cast<unsigned long long>(hist_2d[static_cast<std::size_t>(b)]));
        }
        std::fclose(f_hist2);
    }

    return 0;
}

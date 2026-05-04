// compile: nvcc -O3 -std=c++14 -arch=native "Random Walk.cu" -o random_walk -lcurand
// Windows: nvcc -O3 -std=c++14 -arch=sm_75 "Random Walk.cu" -o random_walk.exe -lcurand

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call)                                                                               \
    do {                                                                                             \
        cudaError_t e = (call);                                                                      \
        if (e != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));     \
            exit(1);                                                                                 \
        }                                                                                            \
    } while (0)

__global__ void init_rng(curandStatePhilox4_32_10_t *rng_states, unsigned long long seed,
                         size_t n_walkers) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx >= n_walkers) {
        return;
    }
    curand_init(seed, idx, 0, &rng_states[idx]);
}

__global__ void simulate_1d_gpu(curandStatePhilox4_32_10_t *rng_states, long long *hist,
                                double *sum_pos, double *sum_pos_sq, int *sample_paths,
                                size_t n_walkers, int n_steps, int n_sample) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx >= n_walkers) {
        return;
    }

    curandStatePhilox4_32_10_t state = rng_states[idx];
    int pos = 0;

    for (int step = 0; step < n_steps; ++step) {
        int move = (curand_uniform(&state) <= 0.5f) ? -1 : 1;
        pos += move;
        atomicAdd(&sum_pos[step], (double)pos);
        atomicAdd(&sum_pos_sq[step], (double)pos * (double)pos);
        if (n_sample > 0 && idx < (size_t)n_sample) {
            sample_paths[idx * (size_t)n_steps + (size_t)step] = pos;
        }
    }

    int hist_idx = pos + n_steps;
    atomicAdd(reinterpret_cast<unsigned long long *>(&hist[hist_idx]), 1ULL);
    rng_states[idx] = state;
}

__global__ void simulate_2d_gpu(curandStatePhilox4_32_10_t *rng_states, unsigned long long *hist_r,
                                double *sum_r2, int *x_paths, int *y_paths, size_t n_walkers,
                                int n_steps, int bins, double max_r, int n_sample) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx >= n_walkers) {
        return;
    }

    curandStatePhilox4_32_10_t state = rng_states[idx];
    int x = 0;
    int y = 0;

    for (int step = 0; step < n_steps; ++step) {
        x += (curand_uniform(&state) <= 0.5f) ? -1 : 1;
        y += (curand_uniform(&state) <= 0.5f) ? -1 : 1;
        atomicAdd(&sum_r2[step], double(x * x + y * y));
        if (n_sample > 0 && idx < (size_t)n_sample) {
            size_t row = idx * (size_t)n_steps + (size_t)step;
            x_paths[row] = x;
            y_paths[row] = y;
        }
    }

    double r = sqrt(double(x * x + y * y));
    int bin = (int)(r / max_r * (double)bins);
    if (bin < 0) {
        bin = 0;
    }
    if (bin >= bins) {
        bin = bins - 1;
    }
    atomicAdd(&hist_r[bin], 1ULL);
    rng_states[idx] = state;
}

int main(int argc, char **argv) {
    const size_t n_walkers = 1000000ULL;
    const int n_steps = 1000;
    const int bins_2d = 100;
    const int n_sample = 0;
    int threads_per_block = 256;

    if (argc >= 2) {
        threads_per_block = atoi(argv[1]);
    }
    if (threads_per_block <= 0) {
        threads_per_block = 256;
    }

    const int blocks_i =
        (int)((n_walkers + (size_t)threads_per_block - 1) / (size_t)threads_per_block);
    const double max_r = sqrt(2.0 * (double)n_steps);

    curandStatePhilox4_32_10_t *d_rng_1d = nullptr;
    curandStatePhilox4_32_10_t *d_rng_2d = nullptr;
    long long *d_hist_1d = nullptr;
    double *d_sum_1d = nullptr;
    double *d_sum_sq_1d = nullptr;
    unsigned long long *d_hist_2d = nullptr;
    double *d_sum_r2 = nullptr;
    int *d_sample_1d = nullptr;
    int *d_x_paths = nullptr;
    int *d_y_paths = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rng_1d, n_walkers * sizeof(curandStatePhilox4_32_10_t)));
    CUDA_CHECK(cudaMalloc(&d_rng_2d, n_walkers * sizeof(curandStatePhilox4_32_10_t)));
    CUDA_CHECK(cudaMalloc(&d_hist_1d, (size_t)(2 * n_steps + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_hist_1d, 0, (size_t)(2 * n_steps + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sum_1d, (size_t)n_steps * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum_sq_1d, (size_t)n_steps * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum_1d, 0, (size_t)n_steps * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum_sq_1d, 0, (size_t)n_steps * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hist_2d, (size_t)bins_2d * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_hist_2d, 0, (size_t)bins_2d * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_sum_r2, (size_t)n_steps * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum_r2, 0, (size_t)n_steps * sizeof(double)));

    if (n_sample > 0) {
        CUDA_CHECK(cudaMalloc(&d_sample_1d, (size_t)n_sample * (size_t)n_steps * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_x_paths, (size_t)n_sample * (size_t)n_steps * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_y_paths, (size_t)n_sample * (size_t)n_steps * sizeof(int)));
    }

    const unsigned long long seed0 = (unsigned long long)time(nullptr);
    init_rng<<<blocks_i, threads_per_block>>>(d_rng_1d, seed0 + 1ULL, n_walkers);
    init_rng<<<blocks_i, threads_per_block>>>(d_rng_2d, seed0 + 2ULL, n_walkers);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));

    {
        simulate_1d_gpu<<<blocks_i, threads_per_block>>>(
            d_rng_1d, d_hist_1d, d_sum_1d, d_sum_sq_1d, d_sample_1d, n_walkers, n_steps, n_sample);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    {
        simulate_2d_gpu<<<blocks_i, threads_per_block>>>(d_rng_2d, d_hist_2d, d_sum_r2, d_x_paths,
                                                         d_y_paths, n_walkers, n_steps, bins_2d,
                                                         max_r, n_sample);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    printf("{\"execution_time_ms\": %.6f}\n", (double)elapsed_ms);

    std::vector<double> sum_gpu((size_t)n_steps);
    std::vector<double> sumsq_gpu((size_t)n_steps);
    std::vector<long long> hist_1d((size_t)(2 * n_steps + 1));
    CUDA_CHECK(cudaMemcpy(sum_gpu.data(), d_sum_1d, (size_t)n_steps * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sumsq_gpu.data(), d_sum_sq_1d, (size_t)n_steps * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hist_1d.data(), d_hist_1d,
                          (size_t)(2 * n_steps + 1) * sizeof(long long), cudaMemcpyDeviceToHost));

    std::vector<double> sum_r2((size_t)n_steps);
    std::vector<unsigned long long> hist_2d((size_t)bins_2d);
    CUDA_CHECK(cudaMemcpy(sum_r2.data(), d_sum_r2, (size_t)n_steps * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hist_2d.data(), d_hist_2d,
                          (size_t)bins_2d * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    FILE *f_sigma = fopen("sigma_1d.csv", "w");
    if (f_sigma) {
        fprintf(f_sigma, "step,sigma_empirical,sigma_theory\n");
        for (int step = 0; step < n_steps; ++step) {
            double mean = sum_gpu[(size_t)step] / (double)n_walkers;
            double mean_sq = sumsq_gpu[(size_t)step] / (double)n_walkers;
            double var = mean_sq - mean * mean;
            if (var < 0 && var > -1e-12) {
                var = 0;
            }
            fprintf(f_sigma, "%d,%.10g,%.10g\n", step + 1, sqrt(var), sqrt((double)step + 1.0));
        }
        fclose(f_sigma);
    }

    FILE *f_hist = fopen("histogram_1d.csv", "w");
    if (f_hist) {
        fprintf(f_hist, "position,count\n");
        for (int i = 0; i < 2 * n_steps + 1; ++i) {
            fprintf(f_hist, "%d,%lld\n", i - n_steps, (long long)hist_1d[(size_t)i]);
        }
        fclose(f_hist);
    }

    FILE *f_sigma2 = fopen("sigma_2d.csv", "w");
    if (f_sigma2) {
        fprintf(f_sigma2, "step,sigma_empirical,sigma_theory\n");
        for (int step = 0; step < n_steps; ++step) {
            double mean_r2 = sum_r2[(size_t)step] / (double)n_walkers;
            fprintf(f_sigma2, "%d,%.10g,%.10g\n", step + 1, sqrt(mean_r2),
                    sqrt(2.0 * ((double)step + 1.0)));
        }
        fclose(f_sigma2);
    }

    FILE *f_hist2 = fopen("histogram_2d.csv", "w");
    if (f_hist2) {
        fprintf(f_hist2, "bin,r_min,r_max,count\n");
        for (int b = 0; b < bins_2d; ++b) {
            double rmin = (double)b * max_r / (double)bins_2d;
            double rmax = (double)(b + 1) * max_r / (double)bins_2d;
            fprintf(f_hist2, "%d,%.5f,%.5f,%llu\n", b, rmin, rmax,
                    (unsigned long long)hist_2d[(size_t)b]);
        }
        fclose(f_hist2);
    }

    cudaFree(d_rng_1d);
    cudaFree(d_rng_2d);
    cudaFree(d_hist_1d);
    cudaFree(d_sum_1d);
    cudaFree(d_sum_sq_1d);
    cudaFree(d_hist_2d);
    cudaFree(d_sum_r2);
    if (d_sample_1d) {
        cudaFree(d_sample_1d);
    }
    if (d_x_paths) {
        cudaFree(d_x_paths);
    }
    if (d_y_paths) {
        cudaFree(d_y_paths);
    }

    return 0;
}

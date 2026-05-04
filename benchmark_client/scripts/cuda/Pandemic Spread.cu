// nvcc -O3 -std=c++14 -arch=native "Pandemic Spread.cu" -o pandemic_spread -lcurand

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define SUSCEPTIBLE 0
#define INFECTED 1
#define RECOVERED 2

#define CUDA_CHECK(call)                                                                               \
    do {                                                                                             \
        cudaError_t err = (call);                                                                    \
        if (err != cudaSuccess) {                                                                    \
            fprintf(stderr, "%s:%d CUDA error %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                                  \
        }                                                                                            \
    } while (0)

__global__ void initCurand(curandState *states, unsigned long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        curand_init(seed, idx, 0, &states[idx]);
}

__global__ void epidemicStep(const int *state_in, int *state_out, const int *offsets,
                             const int *neighbors, int N, float lambda_inf, float gamma,
                             curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    int my_state = state_in[idx];
    curandState localState = states[idx];

    if (my_state == RECOVERED) {
        state_out[idx] = RECOVERED;
    } else if (my_state == INFECTED) {
        if (curand_uniform(&localState) < gamma)
            state_out[idx] = RECOVERED;
        else
            state_out[idx] = INFECTED;
    } else {
        bool gets_infected = false;
        int start_idx = offsets[idx];
        int end_idx = offsets[idx + 1];
        for (int i = start_idx; i < end_idx; ++i) {
            int neighbor_id = neighbors[i];
            if (state_in[neighbor_id] == INFECTED) {
                if (curand_uniform(&localState) < lambda_inf) {
                    gets_infected = true;
                    break;
                }
            }
        }
        state_out[idx] = gets_infected ? INFECTED : SUSCEPTIBLE;
    }

    states[idx] = localState;
}

int main() {
    const int N = 10000;
    const int K = 4;
    const float beta = 0.1f;
    const float lambda_inf = 0.3f;
    const float gamma = 0.1f;
    const int max_days = 200;

    std::vector<std::vector<int>> adj_list(static_cast<size_t>(N));
    std::mt19937 gen(42U);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::uniform_int_distribution<int> rand_node(0, N - 1);

    for (int i = 0; i < N; ++i) {
        for (int j = 1; j <= K / 2; ++j) {
            int right = (i + j) % N;
            if (dis(gen) < beta) {
                int new_target = rand_node(gen);
                while (new_target == i)
                    new_target = rand_node(gen);
                adj_list[static_cast<size_t>(i)].push_back(new_target);
                adj_list[static_cast<size_t>(new_target)].push_back(i);
            } else {
                adj_list[static_cast<size_t>(i)].push_back(right);
                adj_list[static_cast<size_t>(right)].push_back(i);
            }
        }
    }

    std::vector<int> h_offsets(static_cast<size_t>(N + 1), 0);
    std::vector<int> h_neighbors;
    h_neighbors.reserve(static_cast<size_t>(N) * static_cast<size_t>(K));
    for (int i = 0; i < N; ++i) {
        h_offsets[static_cast<size_t>(i)] = static_cast<int>(h_neighbors.size());
        for (int neighbor : adj_list[static_cast<size_t>(i)])
            h_neighbors.push_back(neighbor);
    }
    h_offsets[static_cast<size_t>(N)] = static_cast<int>(h_neighbors.size());

    std::vector<int> h_state(static_cast<size_t>(N), SUSCEPTIBLE);
    for (int i = 0; i < 5; ++i)
        h_state[static_cast<size_t>(rand_node(gen))] = INFECTED;

    int *d_state_in = nullptr;
    int *d_state_out = nullptr;
    int *d_offsets = nullptr;
    int *d_neighbors = nullptr;
    curandState *d_states = nullptr;

    CUDA_CHECK(cudaMalloc(&d_state_in, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_state_out, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets, static_cast<size_t>(N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighbors, h_neighbors.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_states, static_cast<size_t>(N) * sizeof(curandState)));

    CUDA_CHECK(cudaMemcpy(d_state_in, h_state.data(), static_cast<size_t>(N) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), static_cast<size_t>(N + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors, h_neighbors.data(), h_neighbors.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    initCurand<<<numBlocks, blockSize>>>(d_states, 777U, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    int *d_in = d_state_in;
    int *d_out = d_state_out;

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int day = 0; day < max_days; ++day) {
        epidemicStep<<<numBlocks, blockSize>>>(d_in, d_out, d_offsets, d_neighbors, N, lambda_inf,
                                              gamma, d_states);
        CUDA_CHECK(cudaGetLastError());
        int *t = d_in;
        d_in = d_out;
        d_out = t;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    cudaFree(d_state_in);
    cudaFree(d_state_out);
    cudaFree(d_offsets);
    cudaFree(d_neighbors);
    cudaFree(d_states);

    printf("{\"execution_time_ms\": %.6f}\n", (double)elapsed_ms);
    return 0;
}

// nvcc -O3 -std=c++14 -arch=native "Zombie Apocalypse.cu" -o zombie_apocalypse -lcurand

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define DEAD 0
#define HUMAN 1
#define ZOMBIE 2

struct Agent {
    int id, type, x, y;
};

#define CUDA_CHECK(call)                                                                               \
    do {                                                                                             \
        cudaError_t err = (call);                                                                    \
        if (err != cudaSuccess) {                                                                    \
            fprintf(stderr, "%s:%d CUDA error %s\n", __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(1);                                                                                  \
        }                                                                                            \
    } while (0)

__global__ void initCurand(curandState *states, unsigned long seed, int max_agents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < max_agents)
        curand_init(seed, idx, 0, &states[idx]);
}

__global__ void moveAgents(Agent *agents, int *grid, int num_agents, int Nx, int Ny,
                           curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents)
        return;
    Agent a = agents[idx];
    if (a.type == DEAD)
        return;

    curandState localState = states[idx];
    int dir = curand(&localState) % 5;

    int nx = a.x;
    int ny = a.y;
    if (dir == 1)
        ny = (a.y - 1 + Ny) % Ny;
    else if (dir == 2)
        ny = (a.y + 1) % Ny;
    else if (dir == 3)
        nx = (a.x - 1 + Nx) % Nx;
    else if (dir == 4)
        nx = (a.x + 1) % Nx;

    int target_idx = ny * Nx + nx;
    int old = atomicCAS(&grid[target_idx], 0, idx + 1);

    if (old == 0 || old == idx + 1) {
        if (dir != 0 && old == 0)
            atomicCAS(&grid[a.y * Nx + a.x], idx + 1, 0);
        agents[idx].x = nx;
        agents[idx].y = ny;
    } else {
        atomicCAS(&grid[a.y * Nx + a.x], 0, idx + 1);
    }
    states[idx] = localState;
}

__global__ void interact(Agent *agents, int *grid, int num_agents, int Nx, int Ny, float p,
                         curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents)
        return;
    Agent me = agents[idx];
    if (me.type != HUMAN)
        return;

    int neighbors[4] = {
        ((me.y - 1 + Ny) % Ny) * Nx + me.x,
        ((me.y + 1) % Ny) * Nx + me.x,
        me.y * Nx + ((me.x - 1 + Nx) % Nx),
        me.y * Nx + ((me.x + 1) % Nx),
    };

    curandState localState = states[idx];
#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        int gid = neighbors[i];
        int n_id = grid[gid] - 1;
        if (n_id >= 0 && agents[n_id].type == ZOMBIE) {
            float r = curand_uniform(&localState);
            if (r < p) {
                agents[n_id].type = DEAD;
                grid[gid] = 0;
            } else {
                agents[idx].type = ZOMBIE;
                break;
            }
        }
    }
    states[idx] = localState;
}

int main() {
    const int Nx = 150;
    const int Ny = 150;
    const int NL = 1500;
    const int NZ = 50;
    const int steps = 500;
    const float p_win = 0.1f;

    const int total_agents = NL + NZ;
    const int grid_size = Nx * Ny;
    if (total_agents > grid_size) {
        fprintf(stderr, "Too many agents for grid.\n");
        return 1;
    }

    std::vector<Agent> h_agents(static_cast<size_t>(total_agents));
    std::vector<int> h_grid(static_cast<size_t>(grid_size), 0);
    std::vector<int> positions(static_cast<size_t>(grid_size));
    for (int i = 0; i < grid_size; ++i)
        positions[static_cast<size_t>(i)] = i;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(positions.begin(), positions.end(), gen);

    for (int i = 0; i < total_agents; ++i) {
        int pos = positions[static_cast<size_t>(i)];
        h_agents[static_cast<size_t>(i)].id = i;
        h_agents[static_cast<size_t>(i)].type = (i < NL) ? HUMAN : ZOMBIE;
        h_agents[static_cast<size_t>(i)].x = pos % Nx;
        h_agents[static_cast<size_t>(i)].y = pos / Nx;
        const int gx = h_agents[static_cast<size_t>(i)].y * Nx + h_agents[static_cast<size_t>(i)].x;
        h_grid[static_cast<size_t>(gx)] = i + 1;
    }

    Agent *d_agents = nullptr;
    int *d_grid = nullptr;
    curandState *d_states = nullptr;
    CUDA_CHECK(cudaMalloc(&d_agents, static_cast<size_t>(total_agents) * sizeof(Agent)));
    CUDA_CHECK(cudaMalloc(&d_grid, static_cast<size_t>(grid_size) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_states, static_cast<size_t>(total_agents) * sizeof(curandState)));

    CUDA_CHECK(cudaMemcpy(d_agents, h_agents.data(), static_cast<size_t>(total_agents) * sizeof(Agent),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid.data(), static_cast<size_t>(grid_size) * sizeof(int),
                          cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int numBlocks = (total_agents + blockSize - 1) / blockSize;
    initCurand<<<numBlocks, blockSize>>>(d_states, 1234U, total_agents);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int t = 0; t < steps; ++t) {
        moveAgents<<<numBlocks, blockSize>>>(d_agents, d_grid, total_agents, Nx, Ny, d_states);
        interact<<<numBlocks, blockSize>>>(d_agents, d_grid, total_agents, Nx, Ny, p_win, d_states);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    cudaFree(d_agents);
    cudaFree(d_grid);
    cudaFree(d_states);

    printf("{\"execution_time_ms\": %.6f}\n", (double)elapsed_ms);
    return 0;
}

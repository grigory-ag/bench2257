#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

#define CHECK(call)                                                           \
    do {                                                                      \
        const cudaError_t error = (call);                                     \
        if (error != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n'; \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

__global__ void normalizeRowKernel(float* A, float* I, int n, int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float pivot = A[k * n + k];
        A[k * n + idx] /= pivot;
        I[k * n + idx] /= pivot;
    }
}

__global__ void eliminateRowsKernel(float* A, float* I, int n, int k) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n && row != k) {
        const float factor = A[row * n + k];
        const float valA = A[k * n + col];
        const float valI = I[k * n + col];
        A[row * n + col] -= factor * valA;
        I[row * n + col] -= factor * valI;
    }
}

static void readMatrix(const char* filename, std::vector<float>& matrix, int& n) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << '\n';
        exit(1);
    }
    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    if (!file || n <= 0) {
        std::cerr << "Error reading matrix size from: " << filename << '\n';
        exit(1);
    }
    matrix.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    file.read(reinterpret_cast<char*>(matrix.data()),
              static_cast<std::streamsize>(matrix.size() * sizeof(float)));
    if (!file) {
        std::cerr << "Error reading matrix data from: " << filename << '\n';
        exit(1);
    }
}

static void writeMatrix(const char* filename, const std::vector<float>& matrix, int n) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error creating file: " << filename << '\n';
        exit(1);
    }
    file.write(reinterpret_cast<const char*>(&n), sizeof(int));
    file.write(reinterpret_cast<const char*>(matrix.data()),
               static_cast<std::streamsize>(matrix.size() * sizeof(float)));
    if (!file) {
        std::cerr << "Error writing matrix to: " << filename << '\n';
        exit(1);
    }
}

int main() {
    int n = 0;
    std::vector<float> h_A;
    readMatrix("T.dat", h_A, n);

    std::vector<float> h_I(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0f);
    for (int i = 0; i < n; ++i) {
        h_I[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)] = 1.0f;
    }

    float* d_A = nullptr;
    float* d_I = nullptr;
    CHECK(cudaMalloc(&d_A, static_cast<std::size_t>(n) * n * sizeof(float)));
    CHECK(cudaMalloc(&d_I, static_cast<std::size_t>(n) * n * sizeof(float)));

    const int blockSize1D = 256;
    const int gridSize1D = (n + blockSize1D - 1) / blockSize1D;

    const dim3 blockSize2D(16, 16);
    const dim3 gridSize2D((n + blockSize2D.x - 1) / blockSize2D.x,
                          (n + blockSize2D.y - 1) / blockSize2D.y);

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    CHECK(cudaEventCreate(&ev_start));
    CHECK(cudaEventCreate(&ev_stop));
    CHECK(cudaEventRecord(ev_start));

    CHECK(cudaMemcpy(d_A, h_A.data(), static_cast<std::size_t>(n) * n * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_I, h_I.data(), static_cast<std::size_t>(n) * n * sizeof(float),
                     cudaMemcpyHostToDevice));

    for (int k = 0; k < n; ++k) {
        normalizeRowKernel<<<gridSize1D, blockSize1D>>>(d_A, d_I, n, k);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        eliminateRowsKernel<<<gridSize2D, blockSize2D>>>(d_A, d_I, n, k);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
    }

    std::vector<float> h_Result(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    CHECK(cudaMemcpy(h_Result.data(), d_I, static_cast<std::size_t>(n) * n * sizeof(float),
                     cudaMemcpyDeviceToHost));

    CHECK(cudaEventRecord(ev_stop));
    CHECK(cudaEventSynchronize(ev_stop));

    float time_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&time_ms, ev_start, ev_stop));

    writeMatrix("M3.dat", h_Result, n);

    CHECK(cudaEventDestroy(ev_start));
    CHECK(cudaEventDestroy(ev_stop));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_I));

    std::cout << "{\"execution_time_ms\": " << time_ms << "}" << std::endl;
    return 0;
}

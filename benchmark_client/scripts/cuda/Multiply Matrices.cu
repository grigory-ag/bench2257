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

__global__ void matrixMultKernel(const float* M1, const float* M2, float* M3, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += M1[row * n + k] * M2[k * n + col];
        }
        M3[row * n + col] = sum;
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
    int n1 = 0;
    int n2 = 0;
    std::vector<float> M1;
    std::vector<float> M2;

    readMatrix("M1.dat", M1, n1);
    readMatrix("M2.dat", M2, n2);

    if (n1 != n2) {
        std::cerr << "Error: Matrix sizes don't match!\n";
        return 1;
    }
    const int n = n1;
    std::vector<float> M3(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));

    float* d_M1 = nullptr;
    float* d_M2 = nullptr;
    float* d_M3 = nullptr;
    CHECK(cudaMalloc(&d_M1, static_cast<std::size_t>(n) * n * sizeof(float)));
    CHECK(cudaMalloc(&d_M2, static_cast<std::size_t>(n) * n * sizeof(float)));
    CHECK(cudaMalloc(&d_M3, static_cast<std::size_t>(n) * n * sizeof(float)));

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    CHECK(cudaEventCreate(&ev_start));
    CHECK(cudaEventCreate(&ev_stop));
    CHECK(cudaEventRecord(ev_start));

    CHECK(cudaMemcpy(d_M1, M1.data(), static_cast<std::size_t>(n) * n * sizeof(float),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_M2, M2.data(), static_cast<std::size_t>(n) * n * sizeof(float),
                     cudaMemcpyHostToDevice));

    const dim3 blockSize(16, 16);
    const dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);
    matrixMultKernel<<<gridSize, blockSize>>>(d_M1, d_M2, d_M3, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(M3.data(), d_M3, static_cast<std::size_t>(n) * n * sizeof(float),
                     cudaMemcpyDeviceToHost));

    CHECK(cudaEventRecord(ev_stop));
    CHECK(cudaEventSynchronize(ev_stop));

    float time_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&time_ms, ev_start, ev_stop));

    writeMatrix("M3.dat", M3, n);

    CHECK(cudaEventDestroy(ev_start));
    CHECK(cudaEventDestroy(ev_stop));
    CHECK(cudaFree(d_M1));
    CHECK(cudaFree(d_M2));
    CHECK(cudaFree(d_M3));

    std::cout << "{\"execution_time_ms\": " << time_ms << "}" << std::endl;
    return 0;
}

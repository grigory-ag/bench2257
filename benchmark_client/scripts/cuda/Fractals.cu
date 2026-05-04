// compile: nvcc -O3 -arch=native -o Fractals Fractals.cu
// or:     nvcc -O3 -arch=sm_86 -o Fractals Fractals.cu

#include <cstdio>
#include <cuda_runtime.h>

#define WIDTH 4000
#define HEIGHT 4000
#define MAX_ITER 1000

#define CHECK(call)                                                                                \
    do {                                                                                           \
        const cudaError_t error = (call);                                                          \
        if (error != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));                         \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

__global__ void mandelbrotKernel(int *d_out, int w, int h, float xmin, float xmax, float ymin,
                                 float ymax) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px < w && py < h) {
        float c_re = xmin + (float)px / (float)w * (xmax - xmin);
        float c_im = ymin + (float)py / (float)h * (ymax - ymin);

        float z_re = 0.0f;
        float z_im = 0.0f;

        int iter = 0;
        for (iter = 0; iter < MAX_ITER; ++iter) {
            float z_re2 = z_re * z_re;
            float z_im2 = z_im * z_im;

            if (z_re2 + z_im2 > 4.0f)
                break;

            z_im = 2.0f * z_re * z_im + c_im;
            z_re = z_re2 - z_im2 + c_re;
        }
        d_out[py * w + px] = iter;
    }
}

__global__ void juliaKernel(int *d_out, int w, int h, float xmin, float xmax, float ymin,
                            float ymax, float c_re_const, float c_im_const) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px < w && py < h) {
        float z_re = xmin + (float)px / (float)w * (xmax - xmin);
        float z_im = ymin + (float)py / (float)h * (ymax - ymin);

        int iter = 0;
        for (iter = 0; iter < MAX_ITER; ++iter) {
            float z_re2 = z_re * z_re;
            float z_im2 = z_im * z_im;

            if (z_re2 + z_im2 > 4.0f)
                break;

            z_im = 2.0f * z_re * z_im + c_im_const;
            z_re = z_re2 - z_im2 + c_re_const;
        }
        d_out[py * w + px] = iter;
    }
}

__global__ void burningShipKernel(int *d_out, int w, int h, float xmin, float xmax, float ymin,
                                  float ymax) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px < w && py < h) {
        float c_re = xmin + (float)px / (float)w * (xmax - xmin);
        float c_im = ymin + (float)py / (float)h * (ymax - ymin);

        float z_re = 0.0f;
        float z_im = 0.0f;

        int iter = 0;
        for (iter = 0; iter < MAX_ITER; ++iter) {
            float z_re2 = z_re * z_re;
            float z_im2 = z_im * z_im;

            if (z_re2 + z_im2 > 4.0f)
                break;

            float next_im = 2.0f * fabsf(z_re) * fabsf(z_im) + c_im;
            float next_re = z_re2 - z_im2 + c_re;

            z_re = next_re;
            z_im = next_im;
        }
        d_out[py * w + px] = iter;
    }
}

int main() {
    int *d_out = nullptr;
    const size_t size = (size_t)WIDTH * (size_t)HEIGHT * sizeof(int);
    CHECK(cudaMalloc(&d_out, size));

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    CHECK(cudaEventCreate(&ev_start));
    CHECK(cudaEventCreate(&ev_stop));

    CHECK(cudaEventRecord(ev_start));
    mandelbrotKernel<<<gridSize, blockSize>>>(d_out, WIDTH, HEIGHT, -2.0f, 1.0f, -1.5f, 1.5f);
    CHECK(cudaGetLastError());
    juliaKernel<<<gridSize, blockSize>>>(d_out, WIDTH, HEIGHT, -1.6f, 1.6f, -1.6f, 1.6f, -0.7f,
                                         0.27015f);
    CHECK(cudaGetLastError());
    burningShipKernel<<<gridSize, blockSize>>>(d_out, WIDTH, HEIGHT, -2.2f, 1.2f, -2.0f, 1.0f);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(ev_stop));
    CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    CHECK(cudaEventDestroy(ev_start));
    CHECK(cudaEventDestroy(ev_stop));

    printf("{\"execution_time_ms\": %.6f}\n", (double)elapsed_ms);

    CHECK(cudaFree(d_out));
    return 0;
}

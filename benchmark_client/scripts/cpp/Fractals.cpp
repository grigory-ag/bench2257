#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int W = 4000;
constexpr int H = 4000;
constexpr int MAX_ITER = 1000;

inline void mandelbrot_row(int py, int *out_row) {
    const float ymin = -1.5f;
    const float ymax = 1.5f;
    const float xmin = -2.0f;
    const float xmax = 1.0f;
    const float c_im = ymin + (float)py / (float)H * (ymax - ymin);

    for (int px = 0; px < W; ++px) {
        const float c_re = xmin + (float)px / (float)W * (xmax - xmin);
        float z_re = 0.0f;
        float z_im = 0.0f;
        int iter = 0;
        for (iter = 0; iter < MAX_ITER; ++iter) {
            const float z_re2 = z_re * z_re;
            const float z_im2 = z_im * z_im;
            if (z_re2 + z_im2 > 4.0f) {
                break;
            }
            z_im = 2.0f * z_re * z_im + c_im;
            z_re = z_re2 - z_im2 + c_re;
        }
        out_row[px] = iter;
    }
}

inline void julia_row(int py, int *out_row) {
    const float xmin = -1.6f;
    const float xmax = 1.6f;
    const float ymin = -1.6f;
    const float ymax = 1.6f;
    const float c_re_k = -0.7f;
    const float c_im_k = 0.27015f;
    const float z_im0 = ymin + (float)py / (float)H * (ymax - ymin);

    for (int px = 0; px < W; ++px) {
        float z_re = xmin + (float)px / (float)W * (xmax - xmin);
        float z_im = z_im0;
        int iter = 0;
        for (iter = 0; iter < MAX_ITER; ++iter) {
            const float z_re2 = z_re * z_re;
            const float z_im2 = z_im * z_im;
            if (z_re2 + z_im2 > 4.0f) {
                break;
            }
            z_im = 2.0f * z_re * z_im + c_im_k;
            z_re = z_re2 - z_im2 + c_re_k;
        }
        out_row[px] = iter;
    }
}

inline void burning_ship_row(int py, int *out_row) {
    const float xmin = -2.2f;
    const float xmax = 1.2f;
    const float ymin = -2.0f;
    const float ymax = 1.0f;
    const float c_im = ymin + (float)py / (float)H * (ymax - ymin);

    for (int px = 0; px < W; ++px) {
        const float c_re = xmin + (float)px / (float)W * (xmax - xmin);
        float z_re = 0.0f;
        float z_im = 0.0f;
        int iter = 0;
        for (iter = 0; iter < MAX_ITER; ++iter) {
            const float z_re2 = z_re * z_re;
            const float z_im2 = z_im * z_im;
            if (z_re2 + z_im2 > 4.0f) {
                break;
            }
            const float next_im = 2.0f * std::fabs(z_re) * std::fabs(z_im) + c_im;
            const float next_re = z_re2 - z_im2 + c_re;
            z_re = next_re;
            z_im = next_im;
        }
        out_row[px] = iter;
    }
}

void run_mandelbrot(std::vector<int> &buf) {
#pragma omp parallel for schedule(dynamic, 8)
    for (int py = 0; py < H; ++py) {
        mandelbrot_row(py, &buf[static_cast<std::size_t>(py) * W]);
    }
}

void run_julia(std::vector<int> &buf) {
#pragma omp parallel for schedule(dynamic, 8)
    for (int py = 0; py < H; ++py) {
        julia_row(py, &buf[static_cast<std::size_t>(py) * W]);
    }
}

void run_burning_ship(std::vector<int> &buf) {
#pragma omp parallel for schedule(dynamic, 8)
    for (int py = 0; py < H; ++py) {
        burning_ship_row(py, &buf[static_cast<std::size_t>(py) * W]);
    }
}

}  // namespace

int main() {
    std::vector<int> buf(static_cast<std::size_t>(W) * static_cast<std::size_t>(H));

    const auto t0 = std::chrono::high_resolution_clock::now();
    run_mandelbrot(buf);
    run_julia(buf);
    run_burning_ship(buf);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    (void)buf[0];
    std::printf("{\"execution_time_ms\": %.6f}\n", elapsed_ms);
    return 0;
}

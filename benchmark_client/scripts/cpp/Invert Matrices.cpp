#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void read_matrix(const char* filename, int& n, std::vector<float>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: File not found\n";
        exit(1);
    }

    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    if (!file || n <= 0) {
        throw std::runtime_error(std::string("Failed to read matrix size from ") + filename);
    }

    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    data.resize(total);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(total * sizeof(float)));
    if (!file) {
        throw std::runtime_error(std::string("Failed to read matrix data from ") + filename);
    }
}

static void write_matrix(const char* filename, int n, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::string("Failed to open ") + filename);
    }

    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    if (data.size() != total) {
        throw std::runtime_error("Invalid output matrix size.");
    }

    file.write(reinterpret_cast<const char*>(&n), sizeof(int));
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(total * sizeof(float)));
}

static bool invert_gauss_jordan(const std::vector<float>& input, int n, std::vector<float>& inverse) {
    const std::size_t n_sz = static_cast<std::size_t>(n);
    const std::size_t cols = n_sz * 2;
    std::vector<float> aug(n_sz * cols, 0.0f);

    for (int i = 0; i < n; ++i) {
        const std::size_t i_row = static_cast<std::size_t>(i) * cols;
        const std::size_t in_row = static_cast<std::size_t>(i) * n_sz;
        for (int j = 0; j < n; ++j) {
            aug[i_row + static_cast<std::size_t>(j)] = input[in_row + static_cast<std::size_t>(j)];
        }
        aug[i_row + n_sz + static_cast<std::size_t>(i)] = 1.0f;
    }

    for (int i = 0; i < n; ++i) {
        int pivot_row = i;
        float pivot_abs = std::fabs(aug[static_cast<std::size_t>(i) * cols + static_cast<std::size_t>(i)]);
        for (int r = i + 1; r < n; ++r) {
            const float candidate =
                std::fabs(aug[static_cast<std::size_t>(r) * cols + static_cast<std::size_t>(i)]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot_row = r;
            }
        }

        if (pivot_abs <= 1e-12f) {
            return false;
        }

        if (pivot_row != i) {
            const std::size_t i_off = static_cast<std::size_t>(i) * cols;
            const std::size_t p_off = static_cast<std::size_t>(pivot_row) * cols;
            for (std::size_t c = 0; c < cols; ++c) {
                const float tmp = aug[i_off + c];
                aug[i_off + c] = aug[p_off + c];
                aug[p_off + c] = tmp;
            }
        }

        const std::size_t i_off = static_cast<std::size_t>(i) * cols;
        const float pivot = aug[i_off + static_cast<std::size_t>(i)];
        for (std::size_t c = 0; c < cols; ++c) {
            aug[i_off + c] /= pivot;
        }

        for (int r = 0; r < n; ++r) {
            if (r == i) {
                continue;
            }
            const std::size_t r_off = static_cast<std::size_t>(r) * cols;
            const float factor = aug[r_off + static_cast<std::size_t>(i)];
            if (factor == 0.0f) {
                continue;
            }
            for (std::size_t c = 0; c < cols; ++c) {
                aug[r_off + c] -= factor * aug[i_off + c];
            }
        }
    }

    inverse.resize(n_sz * n_sz);
    for (int i = 0; i < n; ++i) {
        const std::size_t i_off = static_cast<std::size_t>(i) * cols;
        const std::size_t out_off = static_cast<std::size_t>(i) * n_sz;
        for (int j = 0; j < n; ++j) {
            inverse[out_off + static_cast<std::size_t>(j)] = aug[i_off + n_sz + static_cast<std::size_t>(j)];
        }
    }

    return true;
}

int main() {
    try {
        int n = 0;
        std::vector<float> t;
        read_matrix("T.dat", n, t);

        std::vector<float> inv;
        const auto start = std::chrono::high_resolution_clock::now();
        const bool ok = invert_gauss_jordan(t, n, inv);
        const auto end = std::chrono::high_resolution_clock::now();
        if (!ok) {
            return 1;
        }

        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        write_matrix("M3.dat", n, inv);
        std::cout << "{\"execution_time_ms\": " << elapsed_ms << "}\n";
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

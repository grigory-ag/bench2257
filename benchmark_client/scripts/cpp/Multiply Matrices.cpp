#include <chrono>
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

int main() {
    try {
        int n1 = 0;
        int n2 = 0;
        std::vector<float> a;
        std::vector<float> b;

        read_matrix("M1.dat", n1, a);
        read_matrix("M2.dat", n2, b);
        if (n1 != n2) {
            return 1;
        }

        const int n = n1;
        const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
        std::vector<float> c(total, 0.0f);

        const auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; ++i) {
            float* c_row = c.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
            for (int k = 0; k < n; ++k) {
                const float aik = a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                                    static_cast<std::size_t>(k)];
                const float* b_row = b.data() + static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
                for (int j = 0; j < n; ++j) {
                    c_row[j] += aik * b_row[j];
                }
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();

        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        write_matrix("M3.dat", n, c);
        std::cout << "{\"execution_time_ms\": " << elapsed_ms << "}\n";
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

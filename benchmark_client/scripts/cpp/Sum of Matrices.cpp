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
        std::vector<float> m1;
        std::vector<float> m2;

        read_matrix("M1.dat", n1, m1);
        read_matrix("M2.dat", n2, m2);
        if (n1 != n2) {
            return 1;
        }

        const std::size_t total = static_cast<std::size_t>(n1) * static_cast<std::size_t>(n1);
        std::vector<float> m3(total);

        const auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < total; ++i) {
            m3[i] = m1[i] + m2[i];
        }
        const auto end = std::chrono::high_resolution_clock::now();

        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        write_matrix("M3.dat", n1, m3);
        std::cout << "{\"execution_time_ms\": " << elapsed_ms << "}\n";
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

enum { SUSCEPTIBLE = 0, INFECTED = 1, RECOVERED = 2 };

static void build_small_world(int N, int K, float beta, std::mt19937 &gen,
                             std::vector<std::vector<int>> &adj) {
    adj.assign(static_cast<std::size_t>(N), {});
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::uniform_int_distribution<int> rnode(0, N - 1);
    for (int i = 0; i < N; ++i) {
        for (int j = 1; j <= K / 2; ++j) {
            const int right = (i + j) % N;
            if (uni(gen) < beta) {
                int nt = rnode(gen);
                while (nt == i) {
                    nt = rnode(gen);
                }
                adj[static_cast<std::size_t>(i)].push_back(nt);
                adj[static_cast<std::size_t>(nt)].push_back(i);
            } else {
                adj[static_cast<std::size_t>(i)].push_back(right);
                adj[static_cast<std::size_t>(right)].push_back(i);
            }
        }
    }
}

static void epidemic_step(const std::vector<int> &st_in, std::vector<int> &st_out,
                          const std::vector<std::vector<int>> &adj, int N, float lambda_inf,
                          float gamma, std::mt19937 &gen) {
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    for (int idx = 0; idx < N; ++idx) {
        const int my = st_in[static_cast<std::size_t>(idx)];
        if (my == RECOVERED) {
            st_out[static_cast<std::size_t>(idx)] = RECOVERED;
        } else if (my == INFECTED) {
            if (uni(gen) < gamma) {
                st_out[static_cast<std::size_t>(idx)] = RECOVERED;
            } else {
                st_out[static_cast<std::size_t>(idx)] = INFECTED;
            }
        } else {
            bool gets = false;
            for (int nb : adj[static_cast<std::size_t>(idx)]) {
                if (st_in[static_cast<std::size_t>(nb)] != INFECTED) {
                    continue;
                }
                if (uni(gen) < lambda_inf) {
                    gets = true;
                    break;
                }
            }
            st_out[static_cast<std::size_t>(idx)] = gets ? INFECTED : SUSCEPTIBLE;
        }
    }
}

int main() {
    constexpr int N = 10000;
    constexpr int K = 4;
    constexpr float beta = 0.1f;
    constexpr float lambda_inf = 0.3f;
    constexpr float gamma = 0.1f;
    constexpr int max_days = 200;

    std::mt19937 gen(42u);
    std::vector<std::vector<int>> adj;
    build_small_world(N, K, beta, gen, adj);

    std::vector<int> state(static_cast<std::size_t>(N), SUSCEPTIBLE);
    std::uniform_int_distribution<int> rnode(0, N - 1);
    for (int i = 0; i < 5; ++i) {
        state[static_cast<std::size_t>(rnode(gen))] = INFECTED;
    }

    std::vector<int> nxt(static_cast<std::size_t>(N));

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int d = 0; d < max_days; ++d) {
        epidemic_step(state, nxt, adj, N, lambda_inf, gamma, gen);
        state.swap(nxt);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    (void)state[0];
    std::printf("{\"execution_time_ms\": %.6f}\n", elapsed_ms);
    return 0;
}

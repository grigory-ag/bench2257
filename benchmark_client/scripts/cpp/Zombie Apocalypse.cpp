#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

enum { DEAD = 0, HUMAN = 1, ZOMBIE = 2 };

struct Agent {
    int id;
    int type;
    int x;
    int y;
};

static inline int cas(std::vector<int> &grid, int addr, int cmp, int val) {
    const int old = grid[static_cast<std::size_t>(addr)];
    if (old == cmp) {
        grid[static_cast<std::size_t>(addr)] = val;
    }
    return old;
}

static void simulate_step(std::vector<Agent> &agents, std::vector<int> &grid, int nx, int ny,
                          std::mt19937 &gen) {
    std::uniform_int_distribution<int> dist_dir(0, 4);
    std::uniform_real_distribution<float> dist_u01(0.0f, 1.0f);
    constexpr float p_win = 0.1f;

    const int na = static_cast<int>(agents.size());
    std::vector<int> order(static_cast<std::size_t>(na));
    for (int i = 0; i < na; ++i) {
        order[static_cast<std::size_t>(i)] = i;
    }
    std::shuffle(order.begin(), order.end(), gen);

    for (int k = 0; k < na; ++k) {
        const int idx = order[static_cast<std::size_t>(k)];
        if (agents[static_cast<std::size_t>(idx)].type == DEAD) {
            continue;
        }
        const int ax = agents[static_cast<std::size_t>(idx)].x;
        const int ay = agents[static_cast<std::size_t>(idx)].y;
        int nx2 = ax;
        int ny2 = ay;
        const int d = dist_dir(gen);
        if (d == 1) {
            ny2 = (ay - 1 + ny) % ny;
        } else if (d == 2) {
            ny2 = (ay + 1) % ny;
        } else if (d == 3) {
            nx2 = (ax - 1 + nx) % nx;
        } else if (d == 4) {
            nx2 = (ax + 1) % nx;
        }
        const int target_idx = ny2 * nx + nx2;
        const int old_cell = ay * nx + ax;
        const int old = cas(grid, target_idx, 0, idx + 1);
        if (old == 0 || old == idx + 1) {
            if (d != 0 && old == 0) {
                cas(grid, old_cell, idx + 1, 0);
            }
            agents[static_cast<std::size_t>(idx)].x = nx2;
            agents[static_cast<std::size_t>(idx)].y = ny2;
        } else {
            cas(grid, old_cell, 0, idx + 1);
        }
    }

    std::shuffle(order.begin(), order.end(), gen);
    for (int k = 0; k < na; ++k) {
        const int idx = order[static_cast<std::size_t>(k)];
        if (agents[static_cast<std::size_t>(idx)].type != HUMAN) {
            continue;
        }
        const int mx = agents[static_cast<std::size_t>(idx)].x;
        const int my = agents[static_cast<std::size_t>(idx)].y;
        const int nbr[4] = {
            ((my - 1 + ny) % ny) * nx + mx,
            ((my + 1) % ny) * nx + mx,
            my * nx + ((mx - 1 + nx) % nx),
            my * nx + ((mx + 1) % nx),
        };
        for (int gid : nbr) {
            const int nid = grid[static_cast<std::size_t>(gid)] - 1;
            if (nid >= 0 && agents[static_cast<std::size_t>(nid)].type == ZOMBIE) {
                if (dist_u01(gen) < p_win) {
                    agents[static_cast<std::size_t>(nid)].type = DEAD;
                    grid[static_cast<std::size_t>(gid)] = 0;
                } else {
                    agents[static_cast<std::size_t>(idx)].type = ZOMBIE;
                }
                break;
            }
        }
    }
}

int main() {
    constexpr int NX = 150;
    constexpr int NY = 150;
    constexpr int NL = 1500;
    constexpr int NZ = 50;
    constexpr int STEPS = 500;
    const int total = NL + NZ;
    const int grid_size = NX * NY;

    std::vector<Agent> agents(static_cast<std::size_t>(total));
    std::vector<int> grid(static_cast<std::size_t>(grid_size), 0);
    std::vector<int> positions(static_cast<std::size_t>(grid_size));
    for (int i = 0; i < grid_size; ++i) {
        positions[static_cast<std::size_t>(i)] = i;
    }

    std::mt19937 gen(12345u);
    std::shuffle(positions.begin(), positions.end(), gen);

    for (int i = 0; i < total; ++i) {
        const int p = positions[static_cast<std::size_t>(i)];
        agents[static_cast<std::size_t>(i)].id = i;
        agents[static_cast<std::size_t>(i)].type = (i < NL) ? HUMAN : ZOMBIE;
        agents[static_cast<std::size_t>(i)].x = p % NX;
        agents[static_cast<std::size_t>(i)].y = p / NX;
        grid[static_cast<std::size_t>(agents[static_cast<std::size_t>(i)].y * NX +
                                      agents[static_cast<std::size_t>(i)].x)] = i + 1;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < STEPS; ++t) {
        simulate_step(agents, grid, NX, NY, gen);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    (void)agents[0].type;
    std::printf("{\"execution_time_ms\": %.6f}\n", elapsed_ms);
    return 0;
}

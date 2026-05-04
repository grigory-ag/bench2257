import json
import time

import numpy as np

SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

N = 10_000
K = 4
BETA = 0.1
LAMBDA = 0.3
GAMMA = 0.1
MAX_DAYS = 200


def build_small_world_adj(rng: np.random.Generator) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(N)]
    for i in range(N):
        for j in range(1, K // 2 + 1):
            right = (i + j) % N
            if rng.random() < BETA:
                nt = int(rng.integers(0, N))
                while nt == i:
                    nt = int(rng.integers(0, N))
                adj[i].append(nt)
                adj[nt].append(i)
            else:
                adj[i].append(right)
                adj[right].append(i)
    return adj


def sir_step(state: np.ndarray, adj: list[list[int]], rng: np.random.Generator) -> np.ndarray:
    out = np.empty(N, dtype=np.int32)
    for idx in range(N):
        my = int(state[idx])
        if my == RECOVERED:
            out[idx] = RECOVERED
        elif my == INFECTED:
            if rng.random() < GAMMA:
                out[idx] = RECOVERED
            else:
                out[idx] = INFECTED
        else:
            infected = False
            for nb in adj[idx]:
                if int(state[nb]) != INFECTED:
                    continue
                if rng.random() < LAMBDA:
                    infected = True
                    break
            out[idx] = INFECTED if infected else SUSCEPTIBLE
    return out


def main() -> int:
    rng = np.random.default_rng(42)
    adj = build_small_world_adj(rng)
    state = np.zeros(N, dtype=np.int32)
    for _ in range(5):
        state[int(rng.integers(0, N))] = INFECTED

    t0 = time.perf_counter()
    for _ in range(MAX_DAYS):
        state = sir_step(state, adj, rng)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(json.dumps({"execution_time_ms": elapsed_ms}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

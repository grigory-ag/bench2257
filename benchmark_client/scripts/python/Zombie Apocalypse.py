import json
import time

import numpy as np

DEAD = 0
HUMAN = 1
ZOMBIE = 2

NX = 150
NY = 150
N_HUMANS = 1500
N_ZOMBIES = 50
STEPS = 500
P_WIN = 0.1


def cas(grid: np.ndarray, addr: int, cmp_: int, val: int) -> int:
    old = int(grid[addr])
    if old == cmp_:
        grid[addr] = val
    return old


def run_step(
    agents_type: np.ndarray,
    agents_x: np.ndarray,
    agents_y: np.ndarray,
    grid: np.ndarray,
    rng: np.random.Generator,
) -> None:
    na = agents_type.shape[0]
    dirs = rng.integers(0, 5, size=na, dtype=np.int32)
    order = rng.permutation(na)

    for ii in range(na):
        idx = int(order[ii])
        if agents_type[idx] == DEAD:
            continue

        ay = int(agents_y[idx])
        ax = int(agents_x[idx])
        nx = ax
        ny = ay
        d = int(dirs[idx])
        if d == 1:
            ny = (ay - 1 + NY) % NY
        elif d == 2:
            ny = (ay + 1) % NY
        elif d == 3:
            nx = (ax - 1 + NX) % NX
        elif d == 4:
            nx = (ax + 1) % NX

        target_idx = ny * NX + nx
        old_cell = ay * NX + ax
        old = cas(grid, target_idx, 0, idx + 1)

        if old == 0 or old == idx + 1:
            if d != 0 and old == 0:
                cas(grid, old_cell, idx + 1, 0)
            agents_x[idx] = nx
            agents_y[idx] = ny
        else:
            cas(grid, old_cell, 0, idx + 1)

    interact_order = rng.permutation(na)
    for jj in range(na):
        idx = int(interact_order[jj])
        if agents_type[idx] != HUMAN:
            continue
        me_x = int(agents_x[idx])
        me_y = int(agents_y[idx])
        nbr_idx = np.array(
            [
                ((me_y - 1 + NY) % NY) * NX + me_x,
                ((me_y + 1) % NY) * NX + me_x,
                me_y * NX + ((me_x - 1 + NX) % NX),
                me_y * NX + ((me_x + 1) % NX),
            ],
            dtype=np.int32,
        )
        for gid in nbr_idx:
            g = int(gid)
            n_id = int(grid[g]) - 1
            if n_id >= 0 and agents_type[n_id] == ZOMBIE:
                if rng.random() < P_WIN:
                    agents_type[n_id] = DEAD
                    grid[g] = 0
                else:
                    agents_type[idx] = ZOMBIE
                    break


def main() -> int:
    total = N_HUMANS + N_ZOMBIES
    rng = np.random.default_rng(seed=12345)

    agents_type = np.zeros(total, dtype=np.int32)
    agents_x = np.zeros(total, dtype=np.int32)
    agents_y = np.zeros(total, dtype=np.int32)
    grid = np.zeros(NX * NY, dtype=np.int32)

    positions = rng.permutation(NX * NY)[:total]
    for i in range(total):
        agents_type[i] = HUMAN if i < N_HUMANS else ZOMBIE
        p = int(positions[i])
        agents_x[i] = p % NX
        agents_y[i] = p // NX
        grid[agents_y[i] * NX + agents_x[i]] = i + 1

    t0 = time.perf_counter()
    for _ in range(STEPS):
        run_step(agents_type, agents_x, agents_y, grid, rng)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(json.dumps({"execution_time_ms": elapsed_ms}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

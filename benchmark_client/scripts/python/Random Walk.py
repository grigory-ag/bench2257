import json
import time

import numpy as np

N_WALKERS = 1_000_000
N_STEPS = 1000
N_BINS_2D = 100
# Batched accumulation avoids holding (walkers × steps) matrices for all walkers at once.
_BATCH = 125_000


def run_1d(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sum_pos = np.zeros(N_STEPS, dtype=np.float64)
    sum_pos_sq = np.zeros(N_STEPS, dtype=np.float64)
    hist_len = 2 * N_STEPS + 1
    hist = np.zeros(hist_len, dtype=np.int64)
    start = 0
    while start < N_WALKERS:
        bn = min(_BATCH, N_WALKERS - start)
        rnd = rng.integers(0, 2, size=(bn, N_STEPS), dtype=np.int8)
        moves = np.where(rnd == 0, -1, 1).astype(np.int16, copy=False)
        pos = np.cumsum(moves, axis=1, dtype=np.int32)
        sum_pos += pos.sum(axis=0, dtype=np.float64)
        sum_pos_sq += (pos.astype(np.float64, copy=False) ** 2).sum(axis=0)
        idx = pos[:, -1].astype(np.int64, copy=False) + N_STEPS
        np.add.at(hist, idx, 1)
        start += bn
    return np.stack([sum_pos, sum_pos_sq], axis=0), hist


def run_2d(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sum_r2 = np.zeros(N_STEPS, dtype=np.float64)
    hist = np.zeros(N_BINS_2D, dtype=np.int64)
    max_r = np.sqrt(2.0 * N_STEPS)
    start = 0
    while start < N_WALKERS:
        bn = min(_BATCH, N_WALKERS - start)
        rnd = rng.integers(0, 2, size=(bn, N_STEPS, 2), dtype=np.int8)
        moves_x = np.where(rnd[..., 0] == 0, -1, 1).astype(np.int16, copy=False)
        moves_y = np.where(rnd[..., 1] == 0, -1, 1).astype(np.int16, copy=False)
        x = np.cumsum(moves_x, axis=1, dtype=np.int32)
        y = np.cumsum(moves_y, axis=1, dtype=np.int32)
        x64 = x.astype(np.float64, copy=False)
        y64 = y.astype(np.float64, copy=False)
        sum_r2 += (x64 * x64 + y64 * y64).sum(axis=0)
        xf = x[:, -1].astype(np.float64)
        yf = y[:, -1].astype(np.float64)
        r_final = np.sqrt(xf * xf + yf * yf)
        b = np.minimum((r_final / max_r * N_BINS_2D).astype(np.int64), N_BINS_2D - 1)
        b = np.maximum(b, 0)
        np.add.at(hist, b, 1)
        start += bn
    return sum_r2, hist


def write_sigma_1d(sum_pos: np.ndarray, sum_pos_sq: np.ndarray) -> None:
    mean = sum_pos / N_WALKERS
    mean_sq = sum_pos_sq / N_WALKERS
    var = mean_sq - mean * mean
    var = np.where((var < 0) & (var > -1e-12), 0.0, var)
    sigma = np.sqrt(var)
    steps = np.arange(1, N_STEPS + 1, dtype=np.float64)
    theory = np.sqrt(steps)
    data = np.column_stack([steps, sigma, theory])
    header = "step,sigma_empirical,sigma_theory"
    np.savetxt(
        "sigma_1d.csv",
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%.10g", "%.10g"],
    )


def write_hist_1d(hist: np.ndarray) -> None:
    positions = np.arange(-N_STEPS, N_STEPS + 1, dtype=np.int64)
    data = np.column_stack([positions, hist])
    np.savetxt(
        "histogram_1d.csv",
        data,
        delimiter=",",
        header="position,count",
        comments="",
        fmt=["%d", "%d"],
    )


def write_sigma_2d(sum_r2: np.ndarray) -> None:
    steps = np.arange(1, N_STEPS + 1, dtype=np.float64)
    emp = np.sqrt(sum_r2 / N_WALKERS)
    theory = np.sqrt(2.0 * steps)
    data = np.column_stack([steps, emp, theory])
    np.savetxt(
        "sigma_2d.csv",
        data,
        delimiter=",",
        header="step,sigma_empirical,sigma_theory",
        comments="",
        fmt=["%d", "%.10g", "%.10g"],
    )


def write_hist_2d(hist: np.ndarray) -> None:
    max_r = np.sqrt(2.0 * N_STEPS)
    b = np.arange(N_BINS_2D)
    rmin = b.astype(np.float64) * max_r / N_BINS_2D
    rmax = (b.astype(np.float64) + 1.0) * max_r / N_BINS_2D
    data = np.column_stack([b, rmin, rmax, hist.astype(np.uint64)])
    np.savetxt(
        "histogram_2d.csv",
        data,
        delimiter=",",
        header="bin,r_min,r_max,count",
        comments="",
        fmt=["%d", "%.5f", "%.5f", "%d"],
    )


def main() -> int:
    rng = np.random.default_rng()

    t0 = time.perf_counter()
    agg_1d, hist_1d = run_1d(rng)
    sum_pos = agg_1d[0]
    sum_pos_sq = agg_1d[1]
    sum_r2, hist_2d = run_2d(rng)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(json.dumps({"execution_time_ms": elapsed_ms}, separators=(",", ":")))

    write_sigma_1d(sum_pos, sum_pos_sq)
    write_hist_1d(hist_1d)
    write_sigma_2d(sum_r2)
    write_hist_2d(hist_2d)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

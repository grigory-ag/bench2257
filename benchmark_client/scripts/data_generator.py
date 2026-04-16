#!/usr/bin/env python3
"""
Generate binary matrix files compatible with C++ readMatrix:
  int32 n (native endian), then n*n float32 values row-major.
"""

from __future__ import annotations

import argparse
import array
import random
import struct
import sys
from pathlib import Path


def write_matrix_bin(path: Path, n: int, floats: array.array) -> None:
    if len(floats) != n * n:
        raise ValueError("float count must be n*n")
    with path.open("wb") as f:
        f.write(struct.pack("i", n))
        floats.tofile(f)


def gen_dense_uniform(rng: random.Random, n: int, lo: float, hi: float) -> array.array:
    return array.array("f", (rng.uniform(lo, hi) for _ in range(n * n)))


def gen_tridiagonal_strict_dd(rng: random.Random, n: int) -> array.array:
    """Full n×n row-major tridiagonal matrix with strict diagonal dominance."""
    u = [rng.uniform(0.5, 2.0) for _ in range(n - 1)]  # A[i, i+1]
    l_sub = [rng.uniform(0.5, 2.0) for _ in range(n - 1)]  # A[i, i-1] == l_sub[i-1]

    diag: list[float] = []
    for i in range(n):
        off = 0.0
        if i > 0:
            off += abs(l_sub[i - 1])
        if i < n - 1:
            off += abs(u[i])
        margin = rng.uniform(1.0, 4.0)
        diag.append(off + margin)

    out = array.array("f", [0.0]) * (n * n)
    for i in range(n):
        base = i * n
        if i > 0:
            out[base + i - 1] = l_sub[i - 1]
        out[base + i] = diag[i]
        if i < n - 1:
            out[base + i + 1] = u[i]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Generate M1.dat, M2.dat, T.dat for matrix benchmarks.")
    p.add_argument("--n", type=int, default=4000, help="matrix dimension (default 4000)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path.cwd(),
        help="directory for output .dat files (default: current working directory)",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default: nondeterministic)")
    args = p.parse_args()

    n = args.n
    if n <= 0:
        print("n must be positive", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    m1 = gen_dense_uniform(rng, n, 0.0, 10.0)
    m2 = gen_dense_uniform(rng, n, 0.0, 10.0)
    t = gen_tridiagonal_strict_dd(rng, n)

    write_matrix_bin(out_dir / "M1.dat", n, m1)
    write_matrix_bin(out_dir / "M2.dat", n, m2)
    write_matrix_bin(out_dir / "T.dat", n, t)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

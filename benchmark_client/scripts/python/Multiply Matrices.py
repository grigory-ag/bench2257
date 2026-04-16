import os
import sys
import time

import numpy as np


def read_matrix(path: str) -> tuple[int, np.ndarray]:
    with open(path, "rb") as f:
        n_arr = np.fromfile(f, dtype=np.int32, count=1)
        if n_arr.size != 1:
            raise RuntimeError("Failed to read n from file.")
        n = int(n_arr[0])
        data = np.fromfile(f, dtype=np.float32, count=n * n)
        if data.size != n * n:
            raise RuntimeError("Failed to read matrix data.")
    return n, data.reshape((n, n))


def write_matrix(path: str, n: int, mat: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(np.int32(n).tobytes())
        mat.reshape(-1).astype(np.float32, copy=False).tofile(f)


def main() -> int:
    if not os.path.exists("M1.dat"):
        sys.exit("Error: M1.dat not found in " + os.getcwd())
    if not os.path.exists("M2.dat"):
        sys.exit("Error: M2.dat not found in " + os.getcwd())

    try:
        n1, a = read_matrix("M1.dat")
        n2, b = read_matrix("M2.dat")
        if n1 != n2:
            raise RuntimeError("Matrix sizes don't match.")

        c = np.empty((n1, n1), dtype=np.float32)
        start = time.perf_counter()
        np.matmul(a, b, out=c)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        write_matrix("M3.dat", n1, c)
        print(f'{{"execution_time_ms": {elapsed_ms}}}')
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


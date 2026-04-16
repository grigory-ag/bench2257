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
    if not os.path.exists("T.dat"):
        sys.exit("Error: T.dat not found in " + os.getcwd())

    try:
        n, t = read_matrix("T.dat")
        start = time.perf_counter()
        inv = np.linalg.inv(t).astype(np.float32, copy=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        write_matrix("M3.dat", n, inv)
        print(f'{{"execution_time_ms": {elapsed_ms}}}')
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

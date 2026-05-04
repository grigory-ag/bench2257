import json
import time

import numpy as np

W = H = 4000
MAX_ITER = 1000
ROW_CHUNK = 64


def mandelbrot_chunk(y0: int, y1: int) -> np.ndarray:
    ch = y1 - y0
    px = np.arange(W, dtype=np.float32)[np.newaxis, :]
    yv = np.arange(y0, y1, dtype=np.float32)[:, np.newaxis]
    c_re = -2.0 + px / float(W) * (1.0 - (-2.0))
    c_im = -1.5 + yv / float(H) * (1.5 - (-1.5))
    z_re = np.zeros((ch, W), dtype=np.float32)
    z_im = np.zeros((ch, W), dtype=np.float32)
    out = np.full((ch, W), MAX_ITER, dtype=np.int32)
    active = np.ones((ch, W), dtype=bool)

    for it in range(MAX_ITER):
        zr2 = z_re * z_re
        zi2 = z_im * z_im
        r2 = zr2 + zi2
        escaped = active & (r2 > 4.0)
        out[escaped] = it
        active &= ~escaped
        if not active.any():
            break
        z_im = np.where(active, 2.0 * z_re * z_im + c_im, z_im)
        z_re = np.where(active, zr2 - zi2 + c_re, z_re)
    return out


def julia_chunk(y0: int, y1: int, c_re_const: float, c_im_const: float) -> np.ndarray:
    ch = y1 - y0
    px = np.arange(W, dtype=np.float32)[np.newaxis, :]
    yv = np.arange(y0, y1, dtype=np.float32)[:, np.newaxis]
    z_re = -1.6 + px / float(W) * (1.6 - (-1.6))
    z_im = -1.6 + yv / float(H) * (1.6 - (-1.6))
    out = np.full((ch, W), MAX_ITER, dtype=np.int32)
    active = np.ones((ch, W), dtype=bool)

    for it in range(MAX_ITER):
        zr2 = z_re * z_re
        zi2 = z_im * z_im
        r2 = zr2 + zi2
        escaped = active & (r2 > 4.0)
        out[escaped] = it
        active &= ~escaped
        if not active.any():
            break
        z_im = np.where(active, 2.0 * z_re * z_im + c_im_const, z_im)
        z_re = np.where(active, zr2 - zi2 + c_re_const, z_re)
    return out


def burning_ship_chunk(y0: int, y1: int) -> np.ndarray:
    ch = y1 - y0
    px = np.arange(W, dtype=np.float32)[np.newaxis, :]
    yv = np.arange(y0, y1, dtype=np.float32)[:, np.newaxis]
    c_re = -2.2 + px / float(W) * (1.2 - (-2.2))
    c_im = -2.0 + yv / float(H) * (1.0 - (-2.0))
    z_re = np.zeros((ch, W), dtype=np.float32)
    z_im = np.zeros((ch, W), dtype=np.float32)
    out = np.full((ch, W), MAX_ITER, dtype=np.int32)
    active = np.ones((ch, W), dtype=bool)

    for it in range(MAX_ITER):
        zr2 = z_re * z_re
        zi2 = z_im * z_im
        r2 = zr2 + zi2
        escaped = active & (r2 > 4.0)
        out[escaped] = it
        active &= ~escaped
        if not active.any():
            break
        z_im_new = 2.0 * np.abs(z_re) * np.abs(z_im) + c_im
        z_re_new = zr2 - zi2 + c_re
        z_im = np.where(active, z_im_new, z_im)
        z_re = np.where(active, z_re_new, z_re)
    return out


def run_fractal_row_chunks(chunk_fn, *args) -> None:
    """Compute in-place into one buffer reused across all three fractals (no disk)."""
    buf = np.zeros((H, W), dtype=np.int32)
    for y0 in range(0, H, ROW_CHUNK):
        y1 = min(y0 + ROW_CHUNK, H)
        buf[y0:y1, :] = chunk_fn(y0, y1, *args)


def main() -> int:
    t0 = time.perf_counter()
    run_fractal_row_chunks(mandelbrot_chunk)
    run_fractal_row_chunks(julia_chunk, -0.7, 0.27015)
    run_fractal_row_chunks(burning_ship_chunk)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(json.dumps({"execution_time_ms": elapsed_ms}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

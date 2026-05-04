#!/usr/bin/env python3
"""
Сборка десктоп-утилиты benchmark_client/app.py в один исполняемый файл через PyInstaller.
Запуск из корня репозитория bench2257: python build_client.py
Требуется: pip install pyinstaller (и зависимости из benchmark_client/requirements.txt).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    app_py = root / "benchmark_client" / "app.py"
    dist_dir = root / "benchmark_client" / "dist"
    work_dir = root / "benchmark_client" / "build"

    if not app_py.is_file():
        print(f"Не найден входной файл: {app_py}", file=sys.stderr)
        return 1

    try:
        import customtkinter  # noqa: F401
    except ImportError as e:
        print("Установите customtkinter: pip install -r benchmark_client/requirements.txt", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    ctk_path = os.path.dirname(customtkinter.__file__)
    sep = ";" if sys.platform == "win32" else ":"
    add_data_ctk = f"{ctk_path}{sep}customtkinter"

    dist_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(app_py),
        "--onefile",
        "--name=BenchmarkClient",
        "--windowed",
        f"--distpath={dist_dir}",
        f"--workpath={work_dir}",
        f"--add-data={add_data_ctk}",
        "--hidden-import=cpuinfo",
        "--noconfirm",
    ]

    print("Запуск:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(root))
    if r.returncode != 0:
        print("PyInstaller завершился с ошибкой.", file=sys.stderr)
        return r.returncode

    print("Сборка завершена. Файл BenchmarkClient.exe находится в папке benchmark_client/dist/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

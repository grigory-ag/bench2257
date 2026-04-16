import json
import os
import re
import subprocess
import sys
import tempfile
import time
import threading
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
import psutil
import cpuinfo
import requests

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

API_BASE = "http://localhost:3000"

LANGUAGES = ["CPP", "Python", "CUDA", "Go"]

TASKS = ["Sum of Matrices", "Multiply Matrices", "Invert Matrices", "Random Walk", "Fractals"]

SCRIPTS_DIR = Path(__file__).parent / "scripts"
BENCHMARK_CLIENT_ROOT = Path(__file__).resolve().parent

EXT_MAP: dict[str, str] = {
    "CPP":    ".cpp",
    "Python": ".py",
    "CUDA":   ".cu",
    "Go":     ".go",
}

COMPILER_CHECK_CMDS: dict[str, list[list[str]]] = {
    "CPP":    [["g++", "--version"]],
    "Python": [["python", "--version"], ["python3", "--version"]],
    "CUDA":   [["nvcc", "--version"]],
    "Go":     [["go", "version"]],
}


def _extract_execution_time_ms(stdout: str | None) -> float | None:
    """Parse optional JSON line(s) with \"execution_time_ms\" from benchmark stdout."""
    if not stdout:
        return None

    def from_obj(obj: object) -> float | None:
        if isinstance(obj, dict):
            v = obj.get("execution_time_ms")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return None

    stripped = stdout.strip()
    for line in stripped.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            parsed = from_obj(json.loads(s))
            if parsed is not None:
                return parsed
        except json.JSONDecodeError:
            m = re.search(r'"execution_time_ms"\s*:\s*([-+0-9.eE]+)', s)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass

    try:
        return from_obj(json.loads(stripped))
    except json.JSONDecodeError:
        return None


def _time_ms_from_run(stdout: str | None, wall_ms: float) -> float:
    parsed = _extract_execution_time_ms(stdout)
    if parsed is not None:
        return round(parsed, 2)
    return round(wall_ms, 2)


# ─────────────────────────────────────────────
#  Hardware helpers
# ─────────────────────────────────────────────

def get_cpu_model() -> str:
    try:
        info = cpuinfo.get_cpu_info()
        return info.get("brand_raw", "Unknown CPU")
    except Exception:
        return "Unknown CPU"


def get_total_ram_mb() -> float:
    return psutil.virtual_memory().total / (1024 ** 2)


def get_gpu_name() -> str:
    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if r.returncode != 0:
                return "Unknown"
            output = r.stdout or ""
            lines = [
                line.strip()
                for line in output.splitlines()
                if line.strip() and line.strip() != "Name"
            ]
            return lines[0] if lines else "Unknown"

        if sys.platform == "darwin":
            r = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if r.returncode != 0:
                return "Unknown"
            for line in r.stdout.splitlines():
                s = line.strip()
                if "Chipset Model:" in s:
                    return s.split("Chipset Model:", 1)[-1].strip()
                if s.startswith("Model:") and ":" in s:
                    return s.split(":", 1)[-1].strip()
            return "Unknown"

        r = subprocess.run(
            ["sh", "-c", "lspci | grep -i vga"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().split("\n")[0].strip()

        r2 = subprocess.run(
            ["lshw", "-C", "display"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r2.returncode != 0:
            return "Unknown"
        for line in r2.stdout.splitlines():
            line = line.strip()
            if line.startswith("product:"):
                return line.split(":", 1)[1].strip()
        return "Unknown"
    except Exception:
        return "Unknown"


def find_python_cmd() -> str | None:
    """Return the first working Python executable name, or None."""
    for cmd in ("python", "python3"):
        try:
            r = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue
    return None


# ─────────────────────────────────────────────
#  Login Window
# ─────────────────────────────────────────────

class LoginWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.token: str | None = None

        self.title("Benchmark Client — Login")
        self.geometry("360x280")
        self.resizable(False, False)

        ctk.CTkLabel(self, text="Benchmark Client", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(28, 4))
        ctk.CTkLabel(self, text="Sign in to your account", text_color="gray").pack(pady=(0, 20))

        self._username_entry = ctk.CTkEntry(self, placeholder_text="Username", width=260)
        self._username_entry.pack(pady=6)

        self._password_entry = ctk.CTkEntry(self, placeholder_text="Password", show="●", width=260)
        self._password_entry.pack(pady=6)

        self._status_label = ctk.CTkLabel(self, text="", text_color="#e05c5c")
        self._status_label.pack(pady=(4, 0))

        self._login_btn = ctk.CTkButton(self, text="Login", width=260, command=self._on_login)
        self._login_btn.pack(pady=(10, 0))

        self.bind("<Return>", lambda _: self._on_login())

    # ------------------------------------------------------------------

    def _on_login(self) -> None:
        username = self._username_entry.get().strip()
        password = self._password_entry.get()

        if not username or not password:
            self._status_label.configure(text="Please fill in all fields.")
            return

        self._login_btn.configure(state="disabled", text="Connecting…")
        self._status_label.configure(text="")
        self.update()

        try:
            resp = requests.post(
                f"{API_BASE}/auth/token",
                data={"username": username, "password": password},
                timeout=8,
            )
        except requests.exceptions.ConnectionError:
            self._show_error("Cannot reach the server.\nMake sure Docker Compose is running.")
            return
        except requests.exceptions.Timeout:
            self._show_error("Request timed out.")
            return
        except requests.exceptions.RequestException as exc:
            self._show_error(f"Network error:\n{exc}")
            return
        finally:
            self._login_btn.configure(state="normal", text="Login")

        if resp.status_code == 200:
            self.token = resp.json().get("access_token")
            self.destroy()
        else:
            detail = resp.json().get("detail", "Authentication failed.")
            self._show_error(detail)

    def _show_error(self, message: str) -> None:
        self._status_label.configure(text=message)
        self._login_btn.configure(state="normal", text="Login")


# ─────────────────────────────────────────────
#  Task Card
# ─────────────────────────────────────────────

class TaskCard(ctk.CTkFrame):
    """Single card representing one task for a given language."""

    def __init__(
        self,
        parent,
        language: str,
        task_name: str,
        token: str,
        cpu_model: str,
        total_ram_mb: float,
        gpu_model: str,
    ) -> None:
        super().__init__(parent, corner_radius=10, border_width=1, border_color="#3a3a3a")
        self._language = language
        self._task_name = task_name
        self._token = token
        self._cpu_model = cpu_model
        self._total_ram_mb = total_ram_mb
        self._gpu_model = gpu_model

        # Title row
        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.pack(fill="x", padx=14, pady=(14, 4))

        ctk.CTkLabel(
            title_frame,
            text=task_name,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(side="left")

        ctk.CTkLabel(
            title_frame,
            text=language,
            font=ctk.CTkFont(size=11),
            text_color="#888",
        ).pack(side="right")

        # Status label
        self._status_label = ctk.CTkLabel(self, text="Ready", text_color="gray", font=ctk.CTkFont(size=12))
        self._status_label.pack(anchor="w", padx=14, pady=(0, 10))

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=14, pady=(0, 14))

        ctk.CTkButton(
            btn_frame,
            text="Check Dependencies",
            width=160,
            fg_color="#2b2b2b",
            hover_color="#3a3a3a",
            command=self._check_deps,
        ).pack(side="left", padx=(0, 8))

        self._run_btn = ctk.CTkButton(
            btn_frame,
            text="Run & Submit",
            width=130,
            fg_color="#1f538d",
            hover_color="#14375e",
            command=self._run_and_submit,
        )
        self._run_btn.pack(side="left")

    # ------------------------------------------------------------------

    def _check_deps(self) -> None:
        lang = self._language
        candidates = COMPILER_CHECK_CMDS.get(lang, [])

        for cmd in candidates:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                version_text = (result.stdout.strip() or result.stderr.strip())
                if result.returncode == 0 or version_text:
                    messagebox.showinfo(
                        title=f"Compiler Found — {lang}",
                        message=f"Compiler found:\n{version_text}",
                    )
                    return
            except FileNotFoundError:
                continue
            except Exception:
                continue

        messagebox.showerror(
            title=f"Compiler Not Found — {lang}",
            message=f"Compiler not found! Please install the required toolchain for {lang}.",
        )

    def _run_and_submit(self) -> None:
        self._run_btn.configure(state="disabled")
        self._status_label.configure(text="Running…", text_color="#f0a500")
        threading.Thread(target=self._worker, daemon=True).start()

    # ------------------------------------------------------------------

    def _run_script(self, lang: str, script_path: Path) -> float:
        """Compile (if needed) and run the script. Returns elapsed time in ms."""

        run_cwd = str(BENCHMARK_CLIENT_ROOT)

        if lang == "Python":
            python_cmd = find_python_cmd()
            if python_cmd is None:
                raise RuntimeError("Python interpreter not found. Please install Python 3.")
            start = time.perf_counter()
            result = subprocess.run(
                [python_cmd, str(script_path)],
                cwd=run_cwd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            wall_ms = (time.perf_counter() - start) * 1000
            if result.returncode != 0:
                err_text = (result.stderr or "").strip()
                raise RuntimeError(err_text or f"Script exited with code {result.returncode}")
            return _time_ms_from_run(result.stdout, wall_ms)

        if lang in ("CPP", "CUDA"):
            compiler = "g++" if lang == "CPP" else "nvcc"
            exe_suffix = ".exe" if sys.platform == "win32" else ""
            fd, exe_path = tempfile.mkstemp(suffix=exe_suffix)
            os.close(fd)
            try:
                compile_result = subprocess.run(
                    [compiler, str(script_path), "-O2", "-o", exe_path],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if compile_result.returncode != 0:
                    raise RuntimeError(f"Compilation failed:\n{compile_result.stderr}")
                start = time.perf_counter()
                run_result = subprocess.run(
                    [exe_path],
                    cwd=run_cwd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                wall_ms = (time.perf_counter() - start) * 1000
                if run_result.returncode != 0:
                    err_text = (run_result.stderr or "").strip()
                    raise RuntimeError(err_text or f"Script exited with code {run_result.returncode}")
                return _time_ms_from_run(run_result.stdout, wall_ms)
            finally:
                try:
                    os.unlink(exe_path)
                except OSError:
                    pass

        if lang == "Go":
            exe_suffix = ".exe" if sys.platform == "win32" else ""
            fd, exe_path = tempfile.mkstemp(suffix=exe_suffix)
            os.close(fd)
            try:
                compile_result = subprocess.run(
                    ["go", "build", "-o", exe_path, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if compile_result.returncode != 0:
                    raise RuntimeError(f"Compilation failed:\n{compile_result.stderr}")
                start = time.perf_counter()
                run_result = subprocess.run(
                    [exe_path],
                    cwd=run_cwd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                wall_ms = (time.perf_counter() - start) * 1000
                if run_result.returncode != 0:
                    err_text = (run_result.stderr or "").strip()
                    raise RuntimeError(err_text or f"Script exited with code {run_result.returncode}")
                return _time_ms_from_run(run_result.stdout, wall_ms)
            finally:
                try:
                    os.unlink(exe_path)
                except OSError:
                    pass

        raise ValueError(f"Unknown language: {lang}")

    def _worker(self) -> None:
        try:
            lang = self._language
            ext = EXT_MAP.get(lang, "")
            script_path = SCRIPTS_DIR / lang / f"{self._task_name}{ext}"

            if not script_path.exists():
                self._set_status("File not found", "#e05c5c")
                messagebox.showerror("File Not Found", f"File not found: {script_path}")
                return

            time_ms = self._run_script(lang, script_path)

            payload = {
                "language": self._language.lower(),
                "task_name": self._task_name,
                "time_ms": time_ms,
                "cpu_max_ram_mb": round(self._total_ram_mb, 2),
                "gpu_max_ram_mb": 0.0,
                "cpu_model": self._cpu_model,
                "gpu_model": self._gpu_model,
            }

            resp = requests.post(
                f"{API_BASE}/api/submissions",
                json=payload,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=10,
            )

            if resp.status_code == 200:
                data = resp.json()
                self._set_status(f"Done — {time_ms} ms", "#4caf50")
                messagebox.showinfo(
                    title="Submission Successful",
                    message=(
                        f"Task:        {self._task_name}\n"
                        f"Language:    {lang}\n"
                        f"Time:        {time_ms} ms\n"
                        f"CPU:         {self._cpu_model}\n"
                        f"RAM total:   {self._total_ram_mb:.0f} MB\n\n"
                        f"Saved with ID: {data.get('id')}"
                    ),
                )
            else:
                detail = resp.json().get("detail", resp.text)
                self._set_status("Submit failed", "#e05c5c")
                messagebox.showerror("Submission Error", f"Server returned {resp.status_code}:\n{detail}")

        except RuntimeError as exc:
            self._set_status("Run failed", "#e05c5c")
            messagebox.showerror("Execution Error", str(exc))
            return
        except requests.exceptions.ConnectionError:
            self._set_status("Connection error", "#e05c5c")
            messagebox.showerror("Connection Error", "Cannot reach the server.\nMake sure Docker Compose is running.")
        except requests.exceptions.Timeout:
            self._set_status("Timeout", "#e05c5c")
            messagebox.showerror("Timeout", "The request timed out.")
        except requests.exceptions.RequestException as exc:
            self._set_status("Network error", "#e05c5c")
            messagebox.showerror("Network Error", str(exc))
        finally:
            self.after(0, lambda: self._run_btn.configure(state="normal"))

    def _set_status(self, text: str, color: str) -> None:
        self.after(0, lambda: self._status_label.configure(text=text, text_color=color))


# ─────────────────────────────────────────────
#  Main Application Window
# ─────────────────────────────────────────────

class MainWindow(ctk.CTk):
    def __init__(self, token: str) -> None:
        super().__init__()
        self._token = token

        # Pre-fetch hardware info once (can be slow)
        self._cpu_model = get_cpu_model()
        self._total_ram_mb = get_total_ram_mb()
        self._gpu_model = get_gpu_name()

        self.title("Benchmark Client")
        self.geometry("860x560")
        self.minsize(700, 460)

        self._build_layout()
        self._select_language("CPP")

    # ------------------------------------------------------------------
    #  Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        # ── Sidebar ──────────────────────────────────────────────────
        self._sidebar = ctk.CTkFrame(self, width=180, corner_radius=0)
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)

        ctk.CTkLabel(
            self._sidebar,
            text="Benchmark\nClient",
            font=ctk.CTkFont(size=17, weight="bold"),
        ).pack(pady=(24, 4))

        ctk.CTkLabel(
            self._sidebar,
            text="Select Language",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).pack(pady=(12, 6))

        self._lang_buttons: dict[str, ctk.CTkButton] = {}
        for lang in LANGUAGES:
            btn = ctk.CTkButton(
                self._sidebar,
                text=lang,
                width=140,
                fg_color="transparent",
                hover_color="#2b2b2b",
                command=lambda l=lang: self._select_language(l),
            )
            btn.pack(pady=4)
            self._lang_buttons[lang] = btn

        # Spacer + hardware info at the bottom
        ctk.CTkFrame(self._sidebar, fg_color="transparent").pack(expand=True)
        ctk.CTkLabel(
            self._sidebar,
            text=f"CPU:\n{self._cpu_model[:28]}",
            font=ctk.CTkFont(size=10),
            text_color="#666",
            wraplength=160,
            justify="center",
        ).pack(pady=(0, 16))

        # ── Main area ─────────────────────────────────────────────────
        self._main_area = ctk.CTkFrame(self, fg_color="transparent")
        self._main_area.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        self._header_label = ctk.CTkLabel(
            self._main_area,
            text="",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self._header_label.pack(anchor="w", pady=(4, 16))

        self._cards_frame = ctk.CTkScrollableFrame(self._main_area, fg_color="transparent")
        self._cards_frame.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    #  Language selection
    # ------------------------------------------------------------------

    def _select_language(self, language: str) -> None:
        self._active_language = language

        for lang, btn in self._lang_buttons.items():
            btn.configure(fg_color="#1f538d" if lang == language else "transparent")

        self._header_label.configure(text=f"{language} — Tasks")

        for widget in self._cards_frame.winfo_children():
            widget.destroy()

        for task in TASKS:
            card = TaskCard(
                self._cards_frame,
                language=language,
                task_name=task,
                token=self._token,
                cpu_model=self._cpu_model,
                total_ram_mb=self._total_ram_mb,
                gpu_model=self._gpu_model,
            )
            card.pack(fill="x", pady=8)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    login = LoginWindow()
    login.mainloop()

    token = login.token
    if not token:
        return  # user closed the login window without authenticating

    app = MainWindow(token=token)
    app.mainloop()


if __name__ == "__main__":
    main()

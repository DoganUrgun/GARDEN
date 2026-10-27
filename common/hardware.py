"""Small accelerator preflight shared by the controller and training workers."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from typing import Any

import config


def configure_training_environment() -> None:
    """Enable the persistent JAX cache before JAX is imported by a worker."""
    config.JAX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(config.JAX_CACHE_DIR))
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "true")


def training_worker_environment() -> dict[str, str]:
    """Return the identical explicit backend/cache environment for every worker."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(config.ROOT), str(config.RUNTIME), environment.get("PYTHONPATH", ""))
    )
    environment["JAX_COMPILATION_CACHE_DIR"] = str(config.JAX_CACHE_DIR)
    environment["JAX_ENABLE_COMPILATION_CACHE"] = "true"
    # Request NVIDIA's CUDA backend explicitly rather than probing other GPUs.
    environment["JAX_PLATFORMS"] = config.TRAINING_JAX_PLATFORM
    return environment


def runtime_accelerator() -> dict[str, Any]:
    """Report the backend in the current process after JAX initialization."""
    import jax

    devices = [str(device) for device in jax.devices()]
    return {
        "backend": jax.default_backend(),
        "devices": devices,
        "process_count": int(jax.process_count()),
        "gpu": jax.default_backend() == "gpu" or any("cuda" in item.lower() for item in devices),
    }


def _nvidia_info() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 3:
            rows.append({"name": fields[0], "driver": fields[1], "memory_mib": fields[2]})
    return rows


def probe_accelerator() -> dict[str, Any]:
    """Probe JAX in a short child process so the controller does not reserve VRAM."""
    script = (
        "import json,jax; d=[str(x) for x in jax.devices()]; "
        "print('REWARD_DESIGN_DEVICE='+json.dumps({'backend':jax.default_backend(),"
        "'devices':d,'process_count':jax.process_count(),"
        "'gpu':jax.default_backend()=='gpu' or any('cuda' in x.lower() for x in d)}))"
    )
    environment = os.environ.copy()
    # The controller intentionally pins its own reward-validation JAX process
    # to CPU. The isolated probe must still report the training accelerator.
    environment.pop("JAX_PLATFORMS", None)
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
            cwd=config.ROOT,
        )
        marker = next(
            (line.split("=", 1)[1] for line in result.stdout.splitlines() if line.startswith("REWARD_DESIGN_DEVICE=")),
            None,
        )
        value = json.loads(marker) if marker else {
            "error": (result.stderr or result.stdout or f"probe exited {result.returncode}").strip()
        }
    except Exception as error:
        value = {"error": f"{type(error).__name__}: {error}"}
    value.update(
        {
            "platform": platform.platform(),
            "wsl": "microsoft" in platform.release().lower()
            or "WSL_DISTRO_NAME" in os.environ,
            "nvidia_smi": _nvidia_info(),
        }
    )
    return value


def require_registered_accelerator() -> dict[str, Any]:
    info = probe_accelerator()
    if config.REQUIRE_GPU and not info.get("gpu"):
        if os.name == "nt":
            guidance = (
                "Native Windows JAX cannot use an NVIDIA GPU. Run the project inside "
                "WSL2 Ubuntu after completing the GPU setup in README.md."
            )
        else:
            guidance = (
                "CUDA-enabled JAX is not active. Check nvidia-smi, install "
                "requirements-gpu.txt, and rerun python main.py doctor."
            )
        raise RuntimeError(f"A GPU is required for registered studies. {guidance} Probe: {info}")
    return info

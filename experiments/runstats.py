# Copyright 2025 David Boetius
import platform
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import cpuinfo
import GPUtil
import psutil


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


def machine_and_code_details() -> dict[str, str]:
    """
    Collects and prints relevant experiment statistics:
     * time
     * git commit
     * platform details (includes operation system name)
     * processor name
     * installed memory
     * current memory usage
     * installed swap
     * current swap usage
     * number of GPus
     * for each GPU: name, memory, memory usage.
    """
    memory_stats = psutil.virtual_memory()
    swap_stats = psutil.swap_memory()
    info = {
        "time": datetime.now(timezone.utc),
        "code_version": git_commit(),
        "platform": platform.platform(aliased=True),
        "CPU": {
            "name": cpuinfo.get_cpu_info()['brand_raw'],
            "physical cores": psutil.cpu_count(logical=False),
            "logical cores": psutil.cpu_count(logical=True),
        },
        "memory": {"total": f"{memory_stats.total / (1024**3)} GB", "used": f"{memory_stats.percent}%"},
        "swap": {"total": f"{swap_stats.total / (1024**3)} GB", "used": f"{swap_stats.percent}%"},
    }
    gpus = GPUtil.getGPUs()
    info["GPUs"] = len(gpus)
    for gpu in gpus:
        info[f"GPU {gpu.id}"] = {
            "name": gpu.name,
            "total memory": f"{gpu.memoryTotal / (1024**2)} MB",
            "used memory": f"{100 * gpu.memoryUsed / gpu.memoryTotal:.1f}%",
        }
    return info

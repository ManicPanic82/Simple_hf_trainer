"""Collect Phase 0 hardware/software evidence for the compatibility report.

Read-only: does not install, upgrade or change anything. Run with the
interpreter you want to describe, for example:

    py -3.12 scripts\\collect_environment.py

The output is Markdown on stdout. It never reads or prints Hugging Face
tokens or other credentials.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime


def run(command: list[str], timeout: int = 30) -> str:
    """Run a command and return combined output, or a short error marker."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return f"<not found: {command[0]}>"
    except subprocess.TimeoutExpired:
        return f"<timeout after {timeout}s: {' '.join(command)}>"
    output = (completed.stdout or "").strip()
    error = (completed.stderr or "").strip()
    if completed.returncode != 0:
        detail = error or output
        return f"<exit {completed.returncode}> {detail}"
    return output or "<no output>"


def powershell(expression: str) -> str:
    return run(["powershell", "-NoProfile", "-Command", expression], timeout=60)


def section(title: str) -> None:
    print(f"\n## {title}\n")


def code_block(text: str, language: str = "text") -> None:
    print(f"```{language}\n{text}\n```")


def main() -> int:
    print("# Phase 0 environment evidence")
    print()
    print(f"Collected: {datetime.now(UTC).isoformat(timespec='seconds')}")
    print(f"Host: {platform.node()}")
    print(f"Interpreter: {sys.executable}")
    print(f"Python: {platform.python_version()} ({platform.architecture()[0]})")

    section("Operating system")

    print(run(["cmd", "/c", "ver"]))
    code_block(
        powershell(
            "Get-CimInstance Win32_OperatingSystem | "
            "Select-Object Caption,Version,BuildNumber,OSArchitecture | Format-List"
        )
    )

    section("CPU and system memory")
    code_block(
        powershell(
            "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,"
            "NumberOfLogicalProcessors,MaxClockSpeed | Format-List; "
            "'Total RAM GB: ' + [math]::Round((Get-CimInstance Win32_ComputerSystem)"
            ".TotalPhysicalMemory/1GB,2); "
            "'Free RAM GB: ' + [math]::Round((Get-CimInstance Win32_OperatingSystem)"
            ".FreePhysicalMemory/1MB,2)"
        )
    )

    section("NVIDIA driver and GPU")
    code_block(run(["nvidia-smi"]))
    code_block(
        run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,"
                "memory.free,utilization.gpu,compute_cap",
                "--format=csv",
            ]
        )
    )

    section("CUDA toolkit")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', '<unset>')}")
    code_block(run(["nvcc", "--version"]))

    section("PyTorch")
    torch_probe = r"""
import json
try:
    import torch
except Exception as error:  # noqa: BLE001 - evidence collection
    print(json.dumps({"import_error": f"{type(error).__name__}: {error}"}))
else:
    info = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )
    print(json.dumps(info, indent=2))
"""
    code_block(run([sys.executable, "-c", f"import os;{torch_probe}"], timeout=120), "json")

    section("Relevant installed packages")
    code_block(
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "list",
                "--format=freeze",
            ],
            timeout=120,
        )
    )

    section("Storage")
    code_block(
        powershell(
            "Get-PSDrive -PSProvider FileSystem | "
            "Where-Object {$_.Name -match '^[A-Z]$'} | "
            "Select-Object Name,@{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}},"
            "@{n='TotalGB';e={[math]::Round(($_.Used+$_.Free)/1GB,1)}} | "
            "Format-Table -AutoSize"
        )
    )

    section("Tool availability")
    tools = {}
    for name in (
        "git",
        "nvcc",
        "nvidia-smi",
        "uv",
        "conda",
        "heretic",
        "winget",
    ):
        path = shutil.which(name)
        tools[name] = path or "<not found>"
    print(json.dumps(tools, indent=2))

    section("Environment (allow-listed, secrets excluded)")
    safe_keys = [
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "CUDA_PATH",
        "CUDA_HOME",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "TORCH_CUDA_ARCH_LIST",
    ]
    print(
        json.dumps(
            {key: os.environ.get(key, "<unset>") for key in safe_keys},
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

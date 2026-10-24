"""Verify that the Suzuki PredictAI development environment is set up correctly.

Run from the project root with the virtual environment active:
    python scripts\\verify_setup.py
"""

from __future__ import annotations

import importlib
import struct
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "openpyxl",
    "requests",
    "dotenv",
    "plotly",
    "matplotlib",
    "sklearn",
    "streamlit",
    "pytest",
]

EXPECTED_DIRS = [
    "data/raw",
    "data/processed",
    "dashboard/pages",
    "docs",
    "scripts",
    "src/market_intelligence",
    "tests",
]


def check_python() -> bool:
    """Check interpreter version, bitness, and whether a venv is active."""
    version = sys.version_info
    bits = struct.calcsize("P") * 8
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)

    print(f"Python version : {version.major}.{version.minor}.{version.micro}")
    print(f"Architecture   : {bits}-bit")
    print(f"Interpreter    : {sys.executable}")
    print(f"Virtual env    : {'ACTIVE' if in_venv else 'NOT ACTIVE'}")

    ok = (version.major, version.minor) == (3, 12) and bits == 64 and in_venv
    if not ok:
        print("  [FAIL] Expected 64-bit Python 3.12 inside the project's .venv")
    return ok


def check_packages() -> bool:
    """Try to import each required package and print its version."""
    all_ok = True
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "version unknown")
            print(f"  [ OK ] {name:<12} {version}")
        except ImportError:
            print(f"  [FAIL] {name:<12} not installed")
            all_ok = False
    return all_ok


def check_folders() -> bool:
    """Confirm that the expected project folders exist."""
    root = Path(__file__).resolve().parent.parent
    all_ok = True
    for rel in EXPECTED_DIRS:
        exists = (root / rel).is_dir()
        print(f"  [{' OK ' if exists else 'FAIL'}] {rel}")
        all_ok = all_ok and exists
    return all_ok


def check_gpu_optional() -> None:
    """Report PyTorch/CUDA status if PyTorch is installed (it is not needed yet)."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        print("  PyTorch not installed (expected at this stage).")
        return
    print(f"  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")


def main() -> int:
    print("=== Python ===")
    python_ok = check_python()
    print("\n=== Packages ===")
    packages_ok = check_packages()
    print("\n=== Folders ===")
    folders_ok = check_folders()
    print("\n=== GPU (optional) ===")
    check_gpu_optional()

    print("\n=== Result ===")
    if python_ok and packages_ok and folders_ok:
        print("All Stage 0 checks passed.")
        return 0
    print("Some checks failed. See [FAIL] lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
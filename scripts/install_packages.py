"""Install the platform requirements with pip's resolver as one transaction."""

import platform
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = {
    "Linux": REPO_ROOT / "requirements_ubuntu.txt",
    "Windows": REPO_ROOT / "requirements_windows.txt",
}


def main() -> None:
    system = platform.system()
    try:
        requirements_file = REQUIREMENTS[system]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported operating system: {system}") from exc

    if not requirements_file.is_file():
        raise FileNotFoundError(f"Requirements file not found: {requirements_file}")

    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Python 3.10 is required, found {sys.version}")
    if system == "Windows":
        try:
            __import__("dlib")
        except ImportError as exc:
            raise RuntimeError(
                "dlib is missing; install the documented conda-forge dlib package first"
            ) from exc

    print(f"Installing dependencies from {requirements_file}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()

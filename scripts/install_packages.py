# -*- coding: utf-8 -*-
"""
"""

import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parent
sys.path.insert(0, str(base_path))
sys.path.insert(0, str((base_path.parent / 'src').resolve()))
sys.path.insert(0, str((base_path.parent / 'src' / 'libs').resolve()))
sys.path.insert(0, str((base_path.parent / 'src' / 'libs' / 'utils').resolve()))
sys.path.insert(0, str((base_path.parent / 'src' / 'libs' / 'logger').resolve()))
sys.path = list(dict.fromkeys(sys.path))

import subprocess
import platform
import hydra
from omegaconf import DictConfig

from log import load_logger_config, getLogger
from utils import repo_root, OmegaConf


def install_package(package):
    global logger
    logger.info(f"Installing package: {package}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info(f"Successfully installed: {package}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package}: {e}")


def get_requirements_file():
    global logger
    system = platform.system()
    logger.debug(f"Detected system: {system}")
    if system == "Windows":
        return Path("requirements_windows.txt")
    elif system == "Linux":
        # Assuming Ubuntu for Linux
        return Path("requirements_ubuntu.txt")
    else:
        logger.error(f"Unsupported operating system: {system}")
        raise RuntimeError(f"Unsupported operating system: {system}")


@hydra.main(config_path="../src/configs", config_name="config", version_base=None)
def main(cfg_dict: DictConfig):
    global logger
    load_logger_config(cfg_dict.logger)
    logger = getLogger("Main")
    req_file = Path(repo_root()) / get_requirements_file()
    if not req_file.is_file():
        logger.error(f"Requirements file not found: {req_file}")
        sys.exit(1)

    logger.info(f"Using requirements file: {req_file}")

    with open(req_file, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                logger.debug(f"Skipping line: {line}")
                continue
            install_package(line)

    logger.info("Installation process completed. Check logs for errors.")


if __name__ == "__main__":
    main()

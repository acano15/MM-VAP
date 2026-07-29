#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip install -r "$REPO_DIR/requirements_ubuntu.txt"

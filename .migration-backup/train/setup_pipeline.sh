#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
extract_env="$repo_dir/.venv-mp"

python3 -m venv "$extract_env"
"$extract_env/bin/python" -m pip install --upgrade pip
"$extract_env/bin/python" -m pip install -r "$repo_dir/train/requirements-extract.txt"

echo "Extraction environment ready."
echo "Run: $extract_env/bin/python train/pipeline.py sync --priority clinical --batch-size 100"

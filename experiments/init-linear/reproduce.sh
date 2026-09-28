#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BASE_PYTHON=/home/jy001/micromamba/envs/virtual-cell/bin/python
RUN_ID=init-linear-s01
RESUME=()
if [[ "${1:-}" == "--resume" ]]; then RESUME=(--resume); shift; fi
if [[ $# -ne 0 ]]; then echo "Usage: ./reproduce.sh [--resume]" >&2; exit 2; fi
exec "$BASE_PYTHON" ../../scripts/research.py execute "$RUN_ID" "${RESUME[@]}" -- \
  bash -c 'set -euo pipefail
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    micromamba run -n virtual-cell uv sync --locked --python /home/jy001/micromamba/envs/virtual-cell/bin/python --no-python-downloads
    exec micromamba run -n virtual-cell uv run --locked --python /home/jy001/micromamba/envs/virtual-cell/bin/python --no-python-downloads python -u src/train.py --config configs/s2-h1-s01.json'

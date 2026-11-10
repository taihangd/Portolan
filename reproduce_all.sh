#!/usr/bin/env bash
# ==============================================================================
# Standalone Evaluation Pipeline for Portolan
# Usage:
#   bash reproduce_all.sh --quick   # Fast offline unit tests + quickstart demo
#   bash reproduce_all.sh           # Full benchmark evaluation suite
# ==============================================================================
set -euo pipefail
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

echo "[1/3] Running offline unit test suite..."
if "${PYTHON_BIN}" -c "import pytest" >/dev/null 2>&1; then
  PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" -m pytest -p no:cacheprovider "${SCRIPT_DIR}/tests/test_open_source.py" -v
else
  PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" "${SCRIPT_DIR}/tests/test_open_source.py"
fi

echo "[2/3] Running standalone quickstart synthesis and 6-engine benchmark..."
PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" "${SCRIPT_DIR}/examples/quickstart_benchmark.py"

if [[ "${1:-}" == "--quick" ]]; then
  echo "[Done] Standalone quick verification PASSED."
  exit 0
fi

echo "[3/3] Executing full benchmark suite..."
PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" "${SCRIPT_DIR}/experiments/run_all_experiments.py" "$@"
echo "[Done] Benchmark metrics and CSV tables saved under experiments/results/."

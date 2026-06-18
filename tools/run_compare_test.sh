#!/usr/bin/env bash
# Standard test for the AE feature-extraction comparison framework.
#
# Usage:
#   bash tools/run_compare_test.sh            # full self-test (M1 + deep if torch)
#   bash tools/run_compare_test.sh --fast     # M1 only, seconds
#   bash tools/run_compare_test.sh --install   # pip install deps first, then test
#
# Exits 0 on PASS, 1 on FAIL.
set -euo pipefail

# repo root = parent of this script's dir
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --install)
      echo ">> installing dependencies from tools/requirements.txt"
      "$PYTHON" -m pip install -r tools/requirements.txt
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

echo ">> python: $($PYTHON --version 2>&1)"
echo ">> running framework self-test"
"$PYTHON" tools/test_compare.py "${EXTRA_ARGS[@]}"
status=$?

if [ "$status" -eq 0 ]; then
  echo ""
  echo ">> OPTIONAL: run the full 6-method comparison on synthetic data and"
  echo "   browse the plots/tables it writes:"
  echo "     $PYTHON tools/run_compare.py --synthetic --epochs 40 --out ae_compare_out"
  echo ""
  echo "   or on your own data:"
  echo "     $PYTHON tools/run_compare.py path/to/data.DTA --methods all --denoise wavelet"
fi

exit $status

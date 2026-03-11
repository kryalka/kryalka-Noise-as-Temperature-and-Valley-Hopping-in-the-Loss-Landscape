#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

INTERP_ROOT="${1:-outputs/artifacts/interpolation_trajectory}"
BARRIER_CFG="${2:-configs/eval/barrier.yaml}"
OUT_ROOT="${3:-outputs/artifacts/barrier_trajectory}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG_DIR="outputs/logs"
FAIL_LOG="$LOG_DIR/barrier_failures.log"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
: > "$FAIL_LOG"

if [[ ! -d "$INTERP_ROOT" ]]; then
  echo "[error] interpolation root not found: $INTERP_ROOT"
  exit 1
fi

if [[ ! -f "$BARRIER_CFG" ]]; then
  echo "[error] barrier config not found: $BARRIER_CFG"
  exit 1
fi

RUN_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

shopt -s nullglob
for interp_csv in "$INTERP_ROOT"/interp__*.csv; do
  [[ -f "$interp_csv" ]] || continue

  meta_json="${interp_csv%.csv}.meta.json"
  if [[ ! -f "$meta_json" ]]; then
    echo "[skip] missing meta for $(basename "$interp_csv")"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    continue
  fi

  pair_tag="$(basename "$interp_csv" .csv)"
  pair_tag="${pair_tag#interp__}"
  out_json="$OUT_ROOT/barrier__${pair_tag}.json"

  if [[ -f "$out_json" ]]; then
    echo "[skip] already exists: $out_json"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    continue
  fi

  echo "[run] barrier for $(basename "$interp_csv")"

  if "$PYTHON_BIN" -m ntempvh.cli barrier \
      --interp_csv "$interp_csv" \
      --config "$BARRIER_CFG" \
      --out "$OUT_ROOT"
  then
    RUN_COUNT=$((RUN_COUNT + 1))
  else
    {
      echo "[fail] barrier for $(basename "$interp_csv")"
      echo "interp_csv=$interp_csv"
      echo "meta_json=$meta_json"
      echo "expected_json=$out_json"
      echo
    } | tee -a "$FAIL_LOG"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done

echo
echo "[done] success: $RUN_COUNT"
echo "[done] skipped: $SKIP_COUNT"
echo "[done] failed : $FAIL_COUNT"
echo "[done] fail log: $FAIL_LOG"
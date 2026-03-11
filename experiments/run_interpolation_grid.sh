#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PAIRS_CSV="${1:-outputs/summaries/trajectory_pairs.csv}"
INTERP_CFG="${2:-configs/eval/interpolation.yaml}"
OUT_ROOT="${3:-outputs/artifacts/interpolation_trajectory}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG_DIR="outputs/logs"
FAIL_LOG="$LOG_DIR/interpolation_failures.log"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
: > "$FAIL_LOG"

if [[ ! -f "$PAIRS_CSV" ]]; then
  echo "[error] trajectory pairs file not found: $PAIRS_CSV"
  exit 1
fi

if [[ ! -f "$INTERP_CFG" ]]; then
  echo "[error] interpolation config not found: $INTERP_CFG"
  exit 1
fi

RUN_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

while IFS=$'\t' read -r run_name epoch_a epoch_b ckpt_a ckpt_b out_csv; do
  [[ -n "${run_name:-}" ]] || continue

  if [[ -f "$out_csv" ]]; then
    echo "[skip] already exists: $run_name epoch_${epoch_a}->epoch_${epoch_b}"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    continue
  fi

  echo "[run] interpolation for $run_name epoch_${epoch_a}->epoch_${epoch_b}"

  if "$PYTHON_BIN" -m ntempvh.cli interpolate \
      --ckptA "$ckpt_a" \
      --ckptB "$ckpt_b" \
      --config "$INTERP_CFG" \
      --out "$OUT_ROOT"
  then
    RUN_COUNT=$((RUN_COUNT + 1))
  else
    {
      echo "[fail] interpolation for $run_name epoch_${epoch_a}->epoch_${epoch_b}"
      echo "ckptA=$ckpt_a"
      echo "ckptB=$ckpt_b"
      echo "out=$out_csv"
      echo
    } | tee -a "$FAIL_LOG"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done < <(
"$PYTHON_BIN" - <<'PY'
import csv
import re
from pathlib import Path

pairs_csv = Path("outputs/summaries/trajectory_pairs.csv")
out_root = Path("outputs/artifacts/interpolation_trajectory")

def lr_to_str(x: float) -> str:
    return f"{x:g}"

with open(pairs_csv, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        run_name = row["run_name"]
        epoch_a = int(row["epoch_A"])
        epoch_b = int(row["epoch_B"])
        ckpt_a = row["ckptA"]
        ckpt_b = row["ckptB"]

        lr = float(row["learning_rate"])
        bs = int(row["batch_size"])
        seed = int(row["seed"])

        out_csv = out_root / (
            f"interp__lr{lr_to_str(lr)}__bs{bs}__seed{seed}"
            f"__e{epoch_a:03d}_e{epoch_b:03d}.csv"
        )

        print(
            "\t".join([
                run_name,
                str(epoch_a),
                str(epoch_b),
                ckpt_a,
                ckpt_b,
                str(out_csv),
            ])
        )
PY
)

echo
echo "[done] success: $RUN_COUNT"
echo "[done] skipped: $SKIP_COUNT"
echo "[done] failed : $FAIL_COUNT"
echo "[done] fail log: $FAIL_LOG"
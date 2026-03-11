#!/usr/bin/env bash
set -euo pipefail

GRID_CONFIG="${1:-configs/train/lr_bs_grid.yaml}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TMP_CFG_DIR=".tmp/lr_bs_grid_configs"
mkdir -p "$TMP_CFG_DIR"

if [[ ! -f "$GRID_CONFIG" ]]; then
  echo "[error] grid config not found: $GRID_CONFIG"
  exit 1
fi

echo "[info] project root: $PROJECT_ROOT"
echo "[info] grid config:  $GRID_CONFIG"
echo "[info] temp cfg dir: $TMP_CFG_DIR"

RUN_COUNT=0
SKIP_DONE_COUNT=0
SKIP_PARTIAL_COUNT=0
PLANNED_COUNT=0


while IFS=$'\t' read -r seed lr bs cfg_path out_root run_dir; do
  PLANNED_COUNT=$((PLANNED_COUNT + 1))

  final_ckpt="${run_dir}/checkpoints/final.pt"

  if [[ -f "$final_ckpt" ]]; then
    echo "[skip-done] seed=${seed} lr=${lr} bs=${bs} -> ${run_dir}"
    SKIP_DONE_COUNT=$((SKIP_DONE_COUNT + 1))
    continue
  fi

  if [[ -d "$run_dir" ]]; then
    echo "[skip-partial] seed=${seed} lr=${lr} bs=${bs} -> ${run_dir}"
    echo "              partial run dir already exists; remove it manually if you want to rerun"
    SKIP_PARTIAL_COUNT=$((SKIP_PARTIAL_COUNT + 1))
    continue
  fi

  echo "[run] seed=${seed} lr=${lr} bs=${bs}"
  ntempvh train --config "$cfg_path" --seed "$seed" --out "$out_root"
  RUN_COUNT=$((RUN_COUNT + 1))

done < <(
python - "$GRID_CONFIG" "$TMP_CFG_DIR" <<'PY'
from __future__ import annotations

import copy
from pathlib import Path
import sys
import yaml

from ntempvh.cli import _format_run_id
from ntempvh.utils.io import load_yaml

grid_config_path = Path(sys.argv[1]).resolve()
tmp_cfg_dir = Path(sys.argv[2]).resolve()
project_root = Path.cwd().resolve()

grid_cfg = load_yaml(grid_config_path)

base_config_path = Path(grid_cfg["base_config"])
if not base_config_path.is_absolute():
    base_config_path = (project_root / base_config_path).resolve()

base_cfg = load_yaml(base_config_path)
out_root = str(grid_cfg.get("out_root", "outputs/runs_lr_bs_grid"))

seeds = list(grid_cfg["seeds"])
learning_rates = list(grid_cfg["learning_rates"])
batch_sizes = list(grid_cfg["batch_sizes"])

tmp_cfg_dir.mkdir(parents=True, exist_ok=True)

def fmt_lr_for_filename(x) -> str:
    s = str(x)
    return s.replace(".", "p").replace("-", "m")

for lr in learning_rates:
    for bs in batch_sizes:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("training", {})
        cfg["training"]["learning_rate"] = float(lr)
        cfg["training"]["batch_size"] = int(bs)

        lr_tag = fmt_lr_for_filename(lr)
        bs_tag = str(bs)

        tmp_cfg_path = tmp_cfg_dir / f"train_lr{lr_tag}_bs{bs_tag}.yaml"
        with open(tmp_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        for seed in seeds:
            run_id = _format_run_id(cfg, int(seed))
            run_dir = Path(out_root) / run_id
            print(
                "\t".join(
                    [
                        str(seed),
                        str(lr),
                        str(bs),
                        str(tmp_cfg_path.relative_to(project_root)),
                        out_root,
                        str(run_dir),
                    ]
                )
            )
PY
)

echo
echo "[done] planned:         $PLANNED_COUNT"
echo "[done] launched:        $RUN_COUNT"
echo "[done] skipped-ready:   $SKIP_DONE_COUNT"
echo "[done] skipped-partial: $SKIP_PARTIAL_COUNT"
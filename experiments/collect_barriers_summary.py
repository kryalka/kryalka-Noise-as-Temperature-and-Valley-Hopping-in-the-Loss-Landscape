#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


PAIRS_CSV = Path("outputs/summaries/trajectory_pairs.csv")
BARRIER_ROOT = Path("outputs/artifacts/barrier_trajectory")
OUT_CSV = Path("outputs/summaries/barriers_trajectory_summary.csv")
OUT_JSON = Path("outputs/summaries/barriers_trajectory_summary.json")


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm_path(p: str) -> str:
    return str(Path(p))


def read_pairs() -> list[dict[str, Any]]:
    with open(PAIRS_CSV, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    if not PAIRS_CSV.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {PAIRS_CSV}")
    if not BARRIER_ROOT.exists():
        raise FileNotFoundError(f"Barrier root not found: {BARRIER_ROOT}")

    pairs = read_pairs()
    pairs_df = pd.DataFrame(pairs)
    pairs_df["ckptA_norm"] = pairs_df["ckptA"].map(norm_path)
    pairs_df["ckptB_norm"] = pairs_df["ckptB"].map(norm_path)

    barrier_rows = []
    unreadable_barriers = []
    missing_meta = []
    unmatched_barriers = []

    for barrier_json in sorted(BARRIER_ROOT.glob("barrier__*.json")):
        try:
            obj = load_json(barrier_json)
        except Exception as e:
            unreadable_barriers.append({
                "barrier_json": str(barrier_json),
                "error": repr(e),
            })
            continue

        interp_csv = obj.get("interp_csv", "")
        if not interp_csv:
            unmatched_barriers.append({
                "barrier_json": str(barrier_json),
                "reason": "missing interp_csv in barrier json",
            })
            continue

        meta_json = Path(interp_csv).with_suffix(".meta.json")
        if not meta_json.exists():
            missing_meta.append({
                "barrier_json": str(barrier_json),
                "interp_csv": interp_csv,
                "expected_meta_json": str(meta_json),
            })
            continue

        try:
            meta = load_json(meta_json)
        except Exception as e:
            unreadable_barriers.append({
                "barrier_json": str(barrier_json),
                "meta_json": str(meta_json),
                "error": repr(e),
            })
            continue

        ckptA = norm_path(meta.get("ckptA", ""))
        ckptB = norm_path(meta.get("ckptB", ""))

        match = pairs_df[
            (pairs_df["ckptA_norm"] == ckptA) &
            (pairs_df["ckptB_norm"] == ckptB)
        ]

        if len(match) == 0:
            unmatched_barriers.append({
                "barrier_json": str(barrier_json),
                "interp_csv": interp_csv,
                "ckptA": ckptA,
                "ckptB": ckptB,
                "reason": "no matching pair in trajectory_pairs.csv",
            })
            continue

        if len(match) > 1:
            unmatched_barriers.append({
                "barrier_json": str(barrier_json),
                "interp_csv": interp_csv,
                "ckptA": ckptA,
                "ckptB": ckptB,
                "reason": "multiple matching pairs in trajectory_pairs.csv",
            })
            continue

        row = match.iloc[0].to_dict()
        jumps = obj.get("jumps", {}) or {}

        barrier_rows.append({
            "run_dir": row["run_dir"],
            "run_name": row["run_name"],
            "dataset": row["dataset"],
            "model": row["model"],
            "seed": int(row["seed"]),
            "learning_rate": float(row["learning_rate"]),
            "batch_size": int(row["batch_size"]),
            "optimizer": row["optimizer"],
            "weight_decay": float(row["weight_decay"]),
            "momentum": float(row["momentum"]),
            "scheduler": row["scheduler"],
            "epochs_total": int(row["epochs_total"]) if row.get("epochs_total", "") != "" else None,
            "epoch_A": int(row["epoch_A"]),
            "epoch_B": int(row["epoch_B"]),
            "ckptA": row["ckptA"],
            "ckptB": row["ckptB"],
            "pair_index": int(row["pair_index"]),
            "interp_csv": obj.get("interp_csv", ""),
            "definition": obj.get("definition", ""),
            "L0": obj.get("L0"),
            "L1": obj.get("L1"),
            "max_L": obj.get("max_L"),
            "peak_t": obj.get("peak_t"),
            "DeltaL": obj.get("DeltaL"),
            "DeltaL_rel": obj.get("DeltaL_rel"),
            "max_diff": obj.get("max_diff"),
            "jump_eps_0.005": jumps.get("0.005"),
            "jump_eps_0.007": jumps.get("0.007"),
            "jump_eps_0.01": jumps.get("0.01"),
            "jump_eps_0.05": jumps.get("0.05"),
            "jump_eps_0.1": jumps.get("0.1"),
            "jump_eps_0.15": jumps.get("0.15"),
            "jump_eps_0.2": jumps.get("0.2"),
            "jump_eps_0.25": jumps.get("0.25"),
            "jump_eps_0.35": jumps.get("0.35"),
            "jump_eps_0.5": jumps.get("0.5"),
            "barrier_json": str(barrier_json),
            "meta_json": str(meta_json),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if barrier_rows:
        out_df = pd.DataFrame(barrier_rows)
        out_df.to_csv(OUT_CSV, index=False)
    else:
        pd.DataFrame().to_csv(OUT_CSV, index=False)

    summary = {
        "pairs_csv": str(PAIRS_CSV),
        "barrier_root": str(BARRIER_ROOT),
        "out_csv": str(OUT_CSV),
        "num_pairs_total": len(pairs_df),
        "num_barriers_matched": len(barrier_rows),
        "num_unreadable_barriers": len(unreadable_barriers),
        "num_missing_meta": len(missing_meta),
        "num_unmatched_barriers": len(unmatched_barriers),
        "unreadable_examples": unreadable_barriers[:20],
        "missing_meta_examples": missing_meta[:20],
        "unmatched_examples": unmatched_barriers[:20],
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved barrier summary: {OUT_CSV}")
    print(f"Saved summary json   : {OUT_JSON}")
    print(f"Pairs total          : {len(pairs_df)}")
    print(f"Barriers matched     : {len(barrier_rows)}")
    print(f"Unreadable barriers  : {len(unreadable_barriers)}")
    print(f"Missing meta         : {len(missing_meta)}")
    print(f"Unmatched barriers   : {len(unmatched_barriers)}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


BARRIERS_CSV = Path("outputs/summaries/barriers_trajectory_summary.csv")
OUT_CSV = Path("outputs/summaries/transitions_trajectory.csv")
OUT_JSON = Path("outputs/summaries/transitions_trajectory_summary.json")

ABS_BARRIER_MIN = 0.05
REL_BARRIER_MIN = 0.10

JUMP_COLUMNS = [
    "jump_eps_0.01",
    "jump_eps_0.05",
    "jump_eps_0.1",
]


# REQUIRE_ALL_JUMPS = True
REQUIRE_ALL_JUMPS = False


def normalize_bool(x: Any) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y"}


def main() -> None:
    if not BARRIERS_CSV.exists():
        raise FileNotFoundError(f"Barrier summary not found: {BARRIERS_CSV}")

    df = pd.read_csv(BARRIERS_CSV)
    if len(df) == 0:
        raise RuntimeError(f"Barrier summary is empty: {BARRIERS_CSV}")

    missing_jump_cols = [c for c in JUMP_COLUMNS if c not in df.columns]
    if missing_jump_cols:
        raise ValueError(f"Missing jump columns in barrier summary: {missing_jump_cols}")

    for col in ["DeltaL", "DeltaL_rel", "max_L", "L0", "L1", "peak_t"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in JUMP_COLUMNS:
        df[col] = df[col].map(normalize_bool)

    df["abs_barrier_ok"] = df["DeltaL"] >= ABS_BARRIER_MIN
    df["rel_barrier_ok"] = df["DeltaL_rel"] >= REL_BARRIER_MIN

    if REQUIRE_ALL_JUMPS:
        df["jump_signal"] = df[JUMP_COLUMNS].all(axis=1)
    else:
        df["jump_signal"] = df[JUMP_COLUMNS].any(axis=1)

    df["is_transition"] = (
        df["abs_barrier_ok"] &
        df["rel_barrier_ok"] &
        df["jump_signal"]
    ).astype(int)

    def first_triggered_jump(row: pd.Series) -> str:
        active = [c for c in JUMP_COLUMNS if bool(row[c])]
        return ",".join(active) if active else ""

    df["triggered_jump_columns"] = df.apply(first_triggered_jump, axis=1)

    def transition_reason(row: pd.Series) -> str:
        reasons = []
        if bool(row["abs_barrier_ok"]):
            reasons.append(f"DeltaL>={ABS_BARRIER_MIN}")
        if bool(row["rel_barrier_ok"]):
            reasons.append(f"DeltaL_rel>={REL_BARRIER_MIN}")
        if bool(row["jump_signal"]):
            reasons.append(f"jump:{row['triggered_jump_columns']}")
        return " | ".join(reasons)

    df["transition_reason"] = df.apply(transition_reason, axis=1)

    sort_cols = [c for c in ["learning_rate", "batch_size", "seed", "epoch_A", "epoch_B"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    summary: dict[str, Any] = {
        "input_csv": str(BARRIERS_CSV),
        "output_csv": str(OUT_CSV),
        "num_pairs_total": int(len(df)),
        "num_transitions": int(df["is_transition"].sum()),
        "num_non_transitions": int((1 - df["is_transition"]).sum()),
        "rule": {
            "abs_barrier_min": ABS_BARRIER_MIN,
            "rel_barrier_min": REL_BARRIER_MIN,
            "jump_columns": JUMP_COLUMNS,
            "require_all_jumps": REQUIRE_ALL_JUMPS,
        },
    }

    if "learning_rate" in df.columns:
        by_lr = (
            df.groupby("learning_rate")["is_transition"]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "num_pairs", "sum": "num_transitions"})
        )
        by_lr["num_non_transitions"] = by_lr["num_pairs"] - by_lr["num_transitions"]
        summary["by_learning_rate"] = by_lr.to_dict(orient="records")

    if all(c in df.columns for c in ["learning_rate", "batch_size"]):
        by_lr_bs = (
            df.groupby(["learning_rate", "batch_size"])["is_transition"]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "num_pairs", "sum": "num_transitions"})
        )
        by_lr_bs["num_non_transitions"] = by_lr_bs["num_pairs"] - by_lr_bs["num_transitions"]
        summary["by_learning_rate_batch_size"] = by_lr_bs.to_dict(orient="records")

    if all(c in df.columns for c in ["run_name"]):
        per_run = (
            df.groupby("run_name")["is_transition"]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "num_pairs", "sum": "num_transitions"})
        )
        per_run["num_non_transitions"] = per_run["num_pairs"] - per_run["num_transitions"]
        summary["per_run"] = per_run.to_dict(orient="records")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved transitions csv: {OUT_CSV}")
    print(f"Saved transitions json: {OUT_JSON}")
    print(f"Pairs total: {len(df)}")
    print(f"Transitions: {int(df['is_transition'].sum())}")
    print(f"Non-transitions: {int((1 - df['is_transition']).sum())}")


if __name__ == "__main__":
    main()
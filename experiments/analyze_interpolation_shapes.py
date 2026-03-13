#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INTERP_ROOT = Path("outputs/artifacts/interpolation_trajectory")
OUT_CSV = Path("outputs/summaries/interpolation_shapes_summary.csv")
OUT_JSON = Path("outputs/summaries/interpolation_shapes_summary.json")

BARRIER_EPS = 0.01
VALLEY_EPS = 0.01

RUN_RE = re.compile(
    r"""
    seed(?P<seed>\d+)
    __opt(?P<optimizer>[^_]+)
    _lr(?P<lr>[^_]+)
    _bs(?P<bs>[^_]+)
    _wd(?P<wd>[^_]+)
    _mom(?P<mom>[^_]+)
    _sch(?P<scheduler>[^_]+)
    __(?P<hash>[a-f0-9]+)
    """,
    re.VERBOSE,
)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_ckpt_path(ckpt_path: str) -> dict[str, Any]:
    p = Path(ckpt_path)
    run_name = p.parent.parent.name

    m = RUN_RE.search(run_name)

    epoch_match = re.match(r"^epoch_(\d+)\.pt$", p.name)

    g = m.groupdict()
    return {
        "run_name": run_name,
        "seed": int(g["seed"]),
        "learning_rate": float(g["lr"]),
        "batch_size": int(g["bs"]),
        "optimizer": g["optimizer"],
        "weight_decay": float(g["wd"]),
        "momentum": float(g["mom"]),
        "scheduler": g["scheduler"],
        "epoch": int(epoch_match.group(1)),
    }


def classify_shape(barrier_height: float, valley_depth: float) -> str:
    barrier_ok = barrier_height >= BARRIER_EPS
    valley_ok = valley_depth >= VALLEY_EPS

    if barrier_ok and not valley_ok:
        return "hump"
    if valley_ok and not barrier_ok:
        return "valley"
    if barrier_ok and valley_ok:
        return "mixed"
    return "flat"


def count_sign_changes(vals: np.ndarray, eps: float = 1e-12) -> int:
    d = np.diff(vals)
    signs = []
    for x in d:
        if x > eps:
            signs.append(1)
        elif x < -eps:
            signs.append(-1)

    if not signs:
        return 0

    changes = 0
    prev = signs[0]
    for s in signs[1:]:
        if s != prev:
            changes += 1
            prev = s
    return changes


def monotonic_label(vals: np.ndarray, eps: float = 1e-12) -> str:
    d = np.diff(vals)
    all_noninc = np.all(d <= eps)
    all_nondec = np.all(d >= -eps)

    if all_noninc:
        return "monotone_decreasing"
    if all_nondec:
        return "monotone_increasing"

    sign_changes = count_sign_changes(vals, eps=eps)
    if sign_changes == 1:
        # U-shape
        first = next((x for x in d if abs(x) > eps), 0.0)
        last = next((x for x in d[::-1] if abs(x) > eps), 0.0)
        if first < 0 and last > 0:
            return "u_shape"
        if first > 0 and last < 0:
            return "hill_shape"

    return "complex"


def main() -> None:
    csv_paths = sorted(INTERP_ROOT.glob("interp__*.csv"))

    rows: list[dict[str, Any]] = []
    bad_files: list[dict[str, str]] = []

    for csv_path in csv_paths:
        meta_path = csv_path.with_suffix(".meta.json")
        if not meta_path.exists():
            bad_files.append({
                "interp_csv": str(csv_path),
                "error": f"Missing meta json: {meta_path}",
            })
            continue

        try:
            meta = load_json(meta_path)
            df = pd.read_csv(csv_path)
        except Exception as e:
            bad_files.append({
                "interp_csv": str(csv_path),
                "error": repr(e),
            })
            continue

        required_cols = {"t", "val_loss", "val_acc"}
        if not required_cols.issubset(df.columns):
            bad_files.append({
                "interp_csv": str(csv_path),
                "error": f"Missing required columns: {required_cols - set(df.columns)}",
            })
            continue

        df = df.sort_values("t").reset_index(drop=True)

        t = df["t"].to_numpy(dtype=float)
        loss = df["val_loss"].to_numpy(dtype=float)
        acc = df["val_acc"].to_numpy(dtype=float)

        L0 = float(loss[0])
        L1 = float(loss[-1])
        A0 = float(acc[0])
        A1 = float(acc[-1])

        baseline = (1.0 - t) * L0 + t * L1
        diff = loss - baseline

        max_idx = int(np.argmax(loss))
        min_idx = int(np.argmin(loss))
        barrier_idx = int(np.argmax(diff))
        valley_idx = int(np.argmin(diff))

        max_L = float(loss[max_idx])
        min_L = float(loss[min_idx])

        barrier_height = float(max(0.0, diff[barrier_idx]))
        valley_depth = float(max(0.0, -diff[valley_idx]))

        peak_t = float(t[barrier_idx])
        valley_t = float(t[valley_idx])
        min_t = float(t[min_idx])
        max_t = float(t[max_idx])

        acc_max = float(np.max(acc))
        acc_min = float(np.min(acc))
        acc_argmax_t = float(t[int(np.argmax(acc))])
        acc_argmin_t = float(t[int(np.argmin(acc))])

        shape = classify_shape(barrier_height, valley_depth)
        mono = monotonic_label(loss)

        info_a = parse_ckpt_path(meta["ckptA"])
        info_b = parse_ckpt_path(meta["ckptB"])

        rows.append({
            "interp_csv": str(csv_path),
            "meta_json": str(meta_path),

            "run_name": info_a["run_name"],
            "seed": info_a["seed"],
            "learning_rate": info_a["learning_rate"],
            "batch_size": info_a["batch_size"],
            "optimizer": info_a["optimizer"],
            "weight_decay": info_a["weight_decay"],
            "momentum": info_a["momentum"],
            "scheduler": info_a["scheduler"],

            "ckptA": meta["ckptA"],
            "ckptB": meta["ckptB"],
            "epoch_A": int(meta.get("epoch_A", info_a["epoch"])),
            "epoch_B": int(meta.get("epoch_B", info_b["epoch"])),

            "num_points": int((meta.get("path") or {}).get("num_points", len(df))),
            "bn_recalib_batches": int((meta.get("path") or {}).get("bn_recalib_batches", 0)),

            "L0": L0,
            "L1": L1,
            "A0": A0,
            "A1": A1,

            "max_L": max_L,
            "max_t": max_t,
            "min_L": min_L,
            "min_t": min_t,

            "baseline_max_diff": barrier_height,
            "baseline_max_diff_t": peak_t,

            "baseline_min_diff": -valley_depth,
            "baseline_min_diff_t": valley_t,
            "valley_depth": valley_depth,

            "endpoints_gap_abs": abs(L1 - L0),
            "endpoints_gap_rel": abs(L1 - L0) / max(abs(L0), abs(L1), 1e-12),

            "endpoint_best_loss": min(L0, L1),
            "endpoint_worst_loss": max(L0, L1),
            "middle_best_improvement_over_best_endpoint": max(0.0, min(L0, L1) - min_L),
            "middle_best_improvement_over_worst_endpoint": max(0.0, max(L0, L1) - min_L),

            "acc_max": acc_max,
            "acc_max_t": acc_argmax_t,
            "acc_min": acc_min,
            "acc_min_t": acc_argmin_t,

            "shape_class": shape,
            "monotonicity_class": mono,
            "num_sign_changes_loss": count_sign_changes(loss),
        })

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    summary: dict[str, Any] = {
        "interp_root": str(INTERP_ROOT),
        "out_csv": str(OUT_CSV),
        "num_pairs_total": int(len(out_df)),
        "num_bad_files": int(len(bad_files)),
        "bad_file_examples": bad_files[:20],
        "barrier_eps": BARRIER_EPS,
        "valley_eps": VALLEY_EPS,
    }

    if len(out_df):
        summary["shape_class_counts"] = (
            out_df["shape_class"].value_counts().sort_index().to_dict()
        )
        summary["monotonicity_class_counts"] = (
            out_df["monotonicity_class"].value_counts().sort_index().to_dict()
        )

        by_epoch = (
            out_df.groupby(["epoch_A", "epoch_B"])["shape_class"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        summary["by_epoch_pair_shape_counts"] = by_epoch.to_dict(orient="records")

        by_lr = (
            out_df.groupby("learning_rate")["shape_class"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        summary["by_learning_rate_shape_counts"] = by_lr.to_dict(orient="records")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved interpolation-shapes csv: {OUT_CSV}")
    print(f"Saved interpolation-shapes json: {OUT_JSON}")
    print(f"Pairs total: {len(out_df)}")
    print(f"Bad files: {len(bad_files)}")

    if len(out_df):
        print("\nShape counts:")
        print(out_df["shape_class"].value_counts())

        print("\nMonotonicity counts:")
        print(out_df["monotonicity_class"].value_counts())


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


PAIRS_CSV = Path("outputs/summaries/trajectory_pairs.csv")
INTERP_ROOT = Path("outputs/artifacts/interpolation_trajectory")
OUT_DONE = Path("outputs/summaries/interpolation_done_from_meta.csv")
OUT_MISSING = Path("outputs/summaries/interpolation_missing.csv")
OUT_SUMMARY = Path("outputs/summaries/interpolation_coverage_summary.json")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm_path(p: str) -> str:
    return str(Path(p))


def main() -> None:
    if not PAIRS_CSV.exists():
        raise FileNotFoundError(f"Missing pairs csv: {PAIRS_CSV}")
    if not INTERP_ROOT.exists():
        raise FileNotFoundError(f"Missing interpolation dir: {INTERP_ROOT}")

    expected = pd.read_csv(PAIRS_CSV)

    meta_rows = []
    for meta_path in sorted(INTERP_ROOT.glob("*.meta.json")):
        try:
            obj = load_json(meta_path)
        except Exception as e:
            meta_rows.append({
                "meta_path": str(meta_path),
                "read_ok": False,
                "error": repr(e),
            })
            continue

        meta_rows.append({
            "meta_path": str(meta_path),
            "read_ok": True,
            "ckptA": norm_path(obj.get("ckptA", "")),
            "ckptB": norm_path(obj.get("ckptB", "")),
            "model": obj.get("model", ""),
            "dataset": obj.get("dataset", ""),
            "epoch_A": obj.get("epoch_A"),
            "epoch_B": obj.get("epoch_B"),
            "seed_A": obj.get("seed_A"),
            "seed_B": obj.get("seed_B"),
            "num_points": ((obj.get("path") or {}).get("num_points")),
            "bn_recalib_batches": ((obj.get("path") or {}).get("bn_recalib_batches")),
            "eval_split": ((obj.get("evaluation") or {}).get("split")),
            "eval_batch_size": ((obj.get("evaluation") or {}).get("batch_size")),
        })

    done_df = pd.DataFrame(meta_rows)
    done_df.to_csv(OUT_DONE, index=False)

    expected["ckptA_norm"] = expected["ckptA"].map(norm_path)
    expected["ckptB_norm"] = expected["ckptB"].map(norm_path)

    done_ok = done_df[done_df["read_ok"] == True].copy()
    if len(done_ok):
        done_ok["ckptA_norm"] = done_ok["ckptA"].map(norm_path)
        done_ok["ckptB_norm"] = done_ok["ckptB"].map(norm_path)
        done_pairs = set(zip(done_ok["ckptA_norm"], done_ok["ckptB_norm"]))
    else:
        done_pairs = set()

    expected["is_done"] = [
        (a, b) in done_pairs
        for a, b in zip(expected["ckptA_norm"], expected["ckptB_norm"])
    ]

    missing_df = expected[expected["is_done"] == False].copy()
    missing_df.to_csv(OUT_MISSING, index=False)

    summary = {
        "pairs_csv": str(PAIRS_CSV),
        "interp_root": str(INTERP_ROOT),
        "num_expected_pairs": int(len(expected)),
        "num_meta_files": int(len(done_df)),
        "num_readable_meta": int((done_df["read_ok"] == True).sum()) if len(done_df) else 0,
        "num_done_pairs": int(expected["is_done"].sum()),
        "num_missing_pairs": int((~expected["is_done"]).sum()),
        "done_csv": str(OUT_DONE),
        "missing_csv": str(OUT_MISSING),
    }

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== INTERPOLATION COVERAGE ===")
    print("expected pairs :", summary["num_expected_pairs"])
    print("meta files     :", summary["num_meta_files"])
    print("readable meta  :", summary["num_readable_meta"])
    print("done pairs     :", summary["num_done_pairs"])
    print("missing pairs  :", summary["num_missing_pairs"])
    print(f"\nSaved done table   : {OUT_DONE}")
    print(f"Saved missing table: {OUT_MISSING}")
    print(f"Saved summary json : {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
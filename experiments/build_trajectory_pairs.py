#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


RUNS_ROOT = Path("outputs/runs_lr_bs_grid")
OUT_CSV = Path("outputs/summaries/trajectory_pairs.csv")
OUT_JSON = Path("outputs/summaries/trajectory_pairs_summary.json")

PAIR_MODE = "milestones"

MILESTONE_EPOCHS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

if PAIR_MODE == "milestones":
    if sorted(set(MILESTONE_EPOCHS)) != MILESTONE_EPOCHS:
        raise ValueError("MILESTONE_EPOCHS must be strictly increasing and unique")

EPOCH_RE = re.compile(r"^epoch_(\d+)\.pt$")
RUN_NAME_RE = re.compile(
    r"""
    ^
    (?P<dataset>[^_]+)_
    (?P<model>[^_]+)_
    seed(?P<seed>\d+)
    __opt(?P<optimizer>[^_]+)
    _lr(?P<lr>[^_]+)
    _bs(?P<bs>[^_]+)
    _wd(?P<wd>[^_]+)
    _mom(?P<mom>[^_]+)
    _sch(?P<scheduler>[^_]+)
    __(?P<hash>[a-f0-9]+)
    $
    """,
    re.VERBOSE,
)


@dataclass
class TrajectoryPair:
    run_dir: str
    run_name: str
    dataset: str
    model: str
    seed: int
    learning_rate: float
    batch_size: int
    optimizer: str
    weight_decay: float
    momentum: float
    scheduler: str
    epochs_total: int
    epoch_A: int
    epoch_B: int
    ckptA: str
    ckptB: str
    pair_index: int


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def parse_run_name(run_name: str) -> dict[str, Any]:
    m = RUN_NAME_RE.match(run_name)
    if not m:
        return {}

    g = m.groupdict()
    out: dict[str, Any] = {
        "dataset": g["dataset"],
        "model": g["model"],
        "seed": safe_int(g["seed"]),
        "optimizer": g["optimizer"],
        "learning_rate": safe_float(g["lr"]),
        "batch_size": safe_int(g["bs"]),
        "weight_decay": safe_float(g["wd"]),
        "momentum": safe_float(g["mom"]),
        "scheduler": g["scheduler"],
    }
    return out


def extract_run_meta(run_dir: Path) -> dict[str, Any]:
    run_name = run_dir.name

    meta: dict[str, Any] = {
        "run_name": run_name,
        "dataset": None,
        "model": None,
        "seed": None,
        "learning_rate": None,
        "batch_size": None,
        "optimizer": None,
        "weight_decay": None,
        "momentum": None,
        "scheduler": None,
        "epochs_total": None,
    }

    parsed = parse_run_name(run_name)
    meta.update({k: v for k, v in parsed.items() if v is not None})

    run_cfg = try_load_json(run_dir / "run_config.json")
    if run_cfg is not None:
        meta["dataset"] = str(run_cfg.get("dataset", meta["dataset"] or ""))
        meta["model"] = str(run_cfg.get("model", meta["model"] or ""))

        training = run_cfg.get("training", {}) or {}
        meta["learning_rate"] = safe_float(training.get("learning_rate", meta["learning_rate"]))
        meta["batch_size"] = safe_int(training.get("batch_size", meta["batch_size"]))
        meta["optimizer"] = str(training.get("optimizer", meta["optimizer"] or ""))
        meta["weight_decay"] = safe_float(training.get("weight_decay", meta["weight_decay"]))
        meta["momentum"] = safe_float(training.get("momentum", meta["momentum"]))
        meta["scheduler"] = str(training.get("scheduler", meta["scheduler"] or ""))
        meta["epochs_total"] = safe_int(training.get("epochs", meta["epochs_total"]))

        seed_val = run_cfg.get("seed", None)
        if seed_val is not None:
            meta["seed"] = safe_int(seed_val)

    manifest = try_load_json(run_dir / "cli_manifest.json")
    if manifest is not None and meta["seed"] is None:
        meta["seed"] = safe_int(manifest.get("seed"))

    summary = try_load_json(run_dir / "summary.json")
    if summary is not None and meta["epochs_total"] is None:
        meta["epochs_total"] = safe_int(summary.get("epochs"))

    missing = [k for k, v in meta.items() if k not in {"run_name"} and v in (None, "")]
    if missing:
        raise ValueError(
            f"Could not fully resolve metadata for run '{run_name}'. "
            f"Missing: {missing}"
        )

    return meta


def collect_epoch_checkpoints(checkpoints_dir: Path) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for path in checkpoints_dir.iterdir():
        if not path.is_file():
            continue
        m = EPOCH_RE.match(path.name)
        if not m:
            continue
        epoch = int(m.group(1))
        pairs.append((epoch, path))

    pairs.sort(key=lambda x: x[0])
    return pairs


def build_trajectory_pairs(run_dir: Path) -> list[TrajectoryPair]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints_dir}")

    meta = extract_run_meta(run_dir)
    epoch_ckpts = collect_epoch_checkpoints(checkpoints_dir)

    if len(epoch_ckpts) < 2:
        raise ValueError(
            f"Run '{run_dir.name}' has fewer than 2 epoch checkpoints; "
            f"found {len(epoch_ckpts)}"
        )

    epoch_to_path = {ep: path for ep, path in epoch_ckpts}
    available_epochs = sorted(epoch_to_path.keys())

    if PAIR_MODE == "adjacent":
        selected_epochs = available_epochs
    elif PAIR_MODE == "milestones":
        selected_epochs = [ep for ep in MILESTONE_EPOCHS if ep in epoch_to_path]
    else:
        raise ValueError(f"Unknown PAIR_MODE: {PAIR_MODE}")

    if len(selected_epochs) < 2:
        raise ValueError(
            f"Run '{run_dir.name}' has fewer than 2 selected epochs under mode={PAIR_MODE}. "
            f"Selected: {selected_epochs}"
        )

    out: list[TrajectoryPair] = []
    for idx in range(len(selected_epochs) - 1):
        epoch_a = selected_epochs[idx]
        epoch_b = selected_epochs[idx + 1]
        ckpt_a = epoch_to_path[epoch_a]
        ckpt_b = epoch_to_path[epoch_b]

        out.append(
            TrajectoryPair(
                run_dir=str(run_dir),
                run_name=meta["run_name"],
                dataset=str(meta["dataset"]),
                model=str(meta["model"]),
                seed=int(meta["seed"]),
                learning_rate=float(meta["learning_rate"]),
                batch_size=int(meta["batch_size"]),
                optimizer=str(meta["optimizer"]),
                weight_decay=float(meta["weight_decay"]),
                momentum=float(meta["momentum"]),
                scheduler=str(meta["scheduler"]),
                epochs_total=int(meta["epochs_total"]),
                epoch_A=epoch_a,
                epoch_B=epoch_b,
                ckptA=str(ckpt_a),
                ckptB=str(ckpt_b),
                pair_index=idx,
            )
        )

    return out


def write_csv(path: Path, rows: Iterable[TrajectoryPair]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(TrajectoryPair.__dataclass_fields__.keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_summary(rows: list[TrajectoryPair]) -> dict[str, Any]:
    runs = sorted({r.run_name for r in rows})
    by_lr: dict[str, int] = {}
    by_lr_bs: dict[str, int] = {}
    by_run: dict[str, int] = {}

    for r in rows:
        lr_key = f"{r.learning_rate:.10g}"
        lr_bs_key = f"lr={r.learning_rate:.10g},bs={r.batch_size}"
        by_lr[lr_key] = by_lr.get(lr_key, 0) + 1
        by_lr_bs[lr_bs_key] = by_lr_bs.get(lr_bs_key, 0) + 1
        by_run[r.run_name] = by_run.get(r.run_name, 0) + 1

    return {
        "runs_root": str(RUNS_ROOT),
        "out_csv": str(OUT_CSV),
        "num_runs": len(runs),
        "num_pairs": len(rows),
        "pairs_per_run": by_run,
        "pairs_by_learning_rate": by_lr,
        "pairs_by_learning_rate_batch_size": by_lr_bs,
        "pair_definition": "consecutive pairs on a predefined milestone grid within one training trajectory",
        "notes": [
            "Pairs are constructed only within a single run.",
            f"PAIR_MODE={PAIR_MODE}",
            f"MILESTONE_EPOCHS={MILESTONE_EPOCHS}",
            "Pairs connect consecutive checkpoints from the selected milestone grid.",
            "best.pt and final.pt are intentionally ignored here.",
        ],
        "pair_mode": PAIR_MODE,
        "milestone_epochs": MILESTONE_EPOCHS if PAIR_MODE == "milestones" else None,
    }


def main() -> None:
    if not RUNS_ROOT.exists():
        raise FileNotFoundError(f"Runs root not found: {RUNS_ROOT}")

    run_dirs = sorted([p for p in RUNS_ROOT.iterdir() if p.is_dir()])
    if not run_dirs:
        raise RuntimeError(f"No run directories found in: {RUNS_ROOT}")

    all_rows: list[TrajectoryPair] = []
    errors: list[str] = []

    for run_dir in run_dirs:
        try:
            rows = build_trajectory_pairs(run_dir)
            all_rows.extend(rows)
        except Exception as e:
            errors.append(f"{run_dir.name}: {e}")

    if errors:
        joined = "\n".join(errors)
        raise RuntimeError(
            "Failed to build trajectory pairs for all runs.\n"
            "Details:\n"
            f"{joined}"
        )

    if not all_rows:
        raise RuntimeError("No trajectory pairs were built.")

    write_csv(OUT_CSV, all_rows)

    summary = build_summary(all_rows)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved trajectory pairs: {OUT_CSV}")
    print(f"Saved summary        : {OUT_JSON}")
    print(f"Runs processed       : {summary['num_runs']}")
    print(f"Pairs collected      : {summary['num_pairs']}")


if __name__ == "__main__":
    main()

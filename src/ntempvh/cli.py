from __future__ import annotations
import argparse
from pathlib import Path

from ntempvh.utils.io import load_yaml, ensure_dir
from ntempvh.train.trainer import train_one_run
from ntempvh.eval.interpolation import run_interpolation
from ntempvh.eval.barrier import compute_barrier
from ntempvh.eval.geometry import compute_geometry

import hashlib
import json

# def _timestamp() -> str:
#     return datetime.now().strftime("%Y%m%d_%H%M%S")

def _short_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

def _format_run_id(cfg: dict, seed: int) -> str:
    tr = cfg.get("training", {}) or {}
    dataset = str(cfg.get("dataset", "data")).lower()
    model = str(cfg.get("model", "model")).lower()

    opt = str(tr.get("optimizer", "sgd")).lower()
    lr = tr.get("learning_rate", "na")
    bs = tr.get("batch_size", "na")
    wd = tr.get("weight_decay", "na")
    mom = tr.get("momentum", "na")
    sch = str(tr.get("scheduler", "none")).lower()

    # h = _short_hash({"cfg": cfg, "seed": int(seed)})
    fingerprint = {
        "dataset": dataset,
        "model": model,
        "training": tr,
        "data_root": cfg.get("data_root"),
        "data": cfg.get("data", {}),
        "seed": int(seed),
    }
    h = _short_hash(fingerprint)

    return f"{dataset}_{model}_seed{seed}__opt{opt}_lr{lr}_bs{bs}_wd{wd}_mom{mom}_sch{sch}__{h}"

def main():
    ap = argparse.ArgumentParser(prog="ntempvh")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train one run")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--out", default="outputs/runs")
    p_train.add_argument("--dry_run", action="store_true")

    p_interp = sub.add_parser("interpolate", help="Interpolate between two checkpoints")
    p_interp.add_argument("--ckptA", required=True)
    p_interp.add_argument("--ckptB", required=True)
    p_interp.add_argument("--config", required=True) 
    # p_interp.add_argument("--out", default="outputs/figures/interp")
    p_interp.add_argument("--out", default="outputs/artifacts/interp")

    p_bar = sub.add_parser("barrier", help="Compute barrier from interpolation csv")
    p_bar.add_argument("--interp_csv", required=True)
    p_bar.add_argument("--config", required=True)  
    # p_bar.add_argument("--out", default="outputs/tables")
    p_bar.add_argument("--out", default="outputs/artifacts/barrier")

    p_geo = sub.add_parser("geometry", help="Compute proxy geometry (curvature) at a checkpoint")
    p_geo.add_argument("--ckpt", required=True)
    p_geo.add_argument("--config", required=True)
    # p_geo.add_argument("--out", default="outputs/tables")
    p_geo.add_argument("--out", default="outputs/artifacts/geometry")

    args = ap.parse_args()

    if args.cmd == "train":
        cfg = load_yaml(args.config)

        train_cfg = cfg.get("training", {})
        if train_cfg.get("batch_size") in (None, "TBD"):
            train_cfg["batch_size"] = 128
        if train_cfg.get("learning_rate") in (None, "TBD"):
            train_cfg["learning_rate"] = 0.1
        cfg["training"] = train_cfg

        # run_id = f"{cfg['dataset']}_{cfg['model']}_seed{args.seed}"
        # base_out = ensure_dir(Path(args.out))
        # run_dir = ensure_dir(base_out / f"{run_id}")

        run_id = _format_run_id(cfg, args.seed)
        base_out = ensure_dir(Path(args.out))
        run_dir = ensure_dir(base_out / run_id)

        manifest = {
            "cmd": "train",
            "config_path": str(args.config),
            "seed": int(args.seed),
            "out_root": str(base_out),
            "run_id": str(run_id),
            "run_dir": str(run_dir),
            "cfg_fingerprint": {
                "dataset": str(cfg.get("dataset", "")),
                "model": str(cfg.get("model", "")),
                "training": cfg.get("training", {}),
                "data_root": cfg.get("data_root"),
                "data": cfg.get("data", {}),
            },
        }
        (run_dir / "cli_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.dry_run:
            print("DRY RUN: created run dir and manifest, training not started.")
            print(f"run dir: {run_dir}")
            return

        ckpt_path = train_one_run(cfg, seed=args.seed, out_dir=str(run_dir))
        print(f"expected metrics: {run_dir / 'metrics.jsonl'}")
        print(f"expected summary: {run_dir / 'summary.json'}")
        print(f"checkpoints dir: {run_dir / 'checkpoints'}")
        print(f"saved checkpoint: {ckpt_path}")
        print(f"run dir: {run_dir}")
        return

    if args.cmd == "interpolate":
        out_dir = str(ensure_dir(Path(args.out)))
        out_csv = run_interpolation(args.ckptA, args.ckptB, args.config, out_dir)
        print(f"saved interpolation: {out_csv}")
        return

    if args.cmd == "barrier":
        out_file = compute_barrier(args.interp_csv, args.config, args.out)
        print(f"saved barrier: {out_file}")
        return
    
    if args.cmd == "geometry":
        out_file = compute_geometry(args.ckpt, args.config, args.out)
        print(f"saved geometry: {out_file}")
        return


if __name__ == "__main__":
    main()

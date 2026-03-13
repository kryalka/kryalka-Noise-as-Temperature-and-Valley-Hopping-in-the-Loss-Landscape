from __future__ import annotations

import json
import re
from pathlib import Path

EXPECTED_LRS = [0.025, 0.05, 0.1, 0.2]
EXPECTED_BS = [32, 64, 128, 256]
EXPECTED_SEEDS = [1, 2]

RUNS_ROOT = Path("outputs/runs_lr_bs_grid")
EXPECT_BEST = True


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_run_name(run_name: str) -> dict | None:
    pattern = re.compile(
        r"^"
        r"(?P<dataset>[^_]+)_(?P<model>[^_]+)_seed(?P<seed>\d+)"
        r"__opt(?P<opt>[^_]+)"
        r"_lr(?P<lr>[^_]+)"
        r"_bs(?P<bs>\d+)"
        r"_wd(?P<wd>[^_]+)"
        r"_mom(?P<mom>[^_]+)"
        r"_sch(?P<sch>[^_]+)"
        r"__(?P<hash>[0-9a-f]+)$"
    )
    m = pattern.match(run_name)
    if m is None:
        return None

    d = m.groupdict()
    return {
        "dataset": d["dataset"],
        "model": d["model"],
        "seed": int(d["seed"]),
        "optimizer": d["opt"],
        "learning_rate": float(d["lr"]),
        "batch_size": int(d["bs"]),
        "weight_decay": d["wd"],
        "momentum": d["mom"],
        "scheduler": d["sch"],
        "hash": d["hash"],
    }


def expected_keys_present(obj: dict, keys: list[str]) -> list[str]:
    return [k for k in keys if k not in obj]


def validate_one_run(run_dir: Path) -> list[str]:
    errors: list[str] = []

    parsed = parse_run_name(run_dir.name)
    if parsed is None:
        return [f"[{run_dir.name}] invalid run dir name format"]

    cli_manifest_path = run_dir / "cli_manifest.json"
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    final_ckpt_path = run_dir / "checkpoints" / "final.pt"
    best_ckpt_path = run_dir / "checkpoints" / "best.pt"

    required_paths = [
        cli_manifest_path,
        run_config_path,
        summary_path,
        metrics_path,
        final_ckpt_path,
    ]
    if EXPECT_BEST:
        required_paths.append(best_ckpt_path)

    for p in required_paths:
        if not p.exists():
            errors.append(f"[{run_dir.name}] missing file: {p}")

    if errors:
        return errors

    try:
        cli_manifest = load_json(cli_manifest_path)
    except Exception as e:
        errors.append(f"[{run_dir.name}] failed to read cli_manifest.json: {e}")
        return errors

    try:
        run_config = load_json(run_config_path)
    except Exception as e:
        errors.append(f"[{run_dir.name}] failed to read run_config.json: {e}")
        return errors

    try:
        summary = load_json(summary_path)
    except Exception as e:
        errors.append(f"[{run_dir.name}] failed to read summary.json: {e}")
        return errors

    try:
        metrics = load_jsonl(metrics_path)
    except Exception as e:
        errors.append(f"[{run_dir.name}] failed to read metrics.jsonl: {e}")
        return errors

    if not metrics:
        errors.append(f"[{run_dir.name}] metrics.jsonl is empty")
        return errors

    manifest_missing = expected_keys_present(
        cli_manifest,
        ["cmd", "seed", "run_id", "run_dir", "cfg_fingerprint"],
    )
    for k in manifest_missing:
        errors.append(f"[{run_dir.name}] cli_manifest missing key: {k}")

    summary_missing = expected_keys_present(
        summary,
        ["seed", "epochs", "final_checkpoint", "final_val_loss", "final_val_acc"],
    )
    for k in summary_missing:
        errors.append(f"[{run_dir.name}] summary missing key: {k}")

    training_cfg = run_config.get("training", {})
    if not isinstance(training_cfg, dict):
        errors.append(f"[{run_dir.name}] run_config.training is not a dict")
        return errors

    if int(parsed["seed"]) != int(cli_manifest.get("seed", -1)):
        errors.append(
            f"[{run_dir.name}] seed mismatch: dir={parsed['seed']} manifest={cli_manifest.get('seed')}"
        )

    if int(parsed["seed"]) != int(run_config.get("seed", -1)):
        errors.append(
            f"[{run_dir.name}] seed mismatch: dir={parsed['seed']} run_config={run_config.get('seed')}"
        )

    if int(parsed["seed"]) != int(summary.get("seed", -1)):
        errors.append(
            f"[{run_dir.name}] seed mismatch: dir={parsed['seed']} summary={summary.get('seed')}"
        )

    rc_lr = float(training_cfg.get("learning_rate"))
    rc_bs = int(training_cfg.get("batch_size"))

    if abs(parsed["learning_rate"] - rc_lr) > 1e-12:
        errors.append(
            f"[{run_dir.name}] learning_rate mismatch: dir={parsed['learning_rate']} run_config={rc_lr}"
        )

    if parsed["batch_size"] != rc_bs:
        errors.append(
            f"[{run_dir.name}] batch_size mismatch: dir={parsed['batch_size']} run_config={rc_bs}"
        )

    epochs = int(training_cfg.get("epochs"))
    summary_epochs = int(summary.get("epochs", -1))
    if epochs != summary_epochs:
        errors.append(
            f"[{run_dir.name}] epochs mismatch: run_config={epochs} summary={summary_epochs}"
        )

    last_row = metrics[-1]
    if "epoch" not in last_row:
        errors.append(f"[{run_dir.name}] last metrics row has no epoch")
    else:
        last_epoch = int(last_row["epoch"])
        if last_epoch != epochs:
            errors.append(
                f"[{run_dir.name}] last metrics epoch mismatch: metrics={last_epoch} expected={epochs}"
            )

    final_checkpoint = str(summary.get("final_checkpoint", ""))
    if not final_checkpoint.endswith("checkpoints/final.pt"):
        errors.append(
            f"[{run_dir.name}] summary.final_checkpoint looks suspicious: {final_checkpoint}"
        )

    required_metric_keys = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    for k in required_metric_keys:
        if k not in last_row:
            errors.append(f"[{run_dir.name}] last metrics row missing key: {k}")

    return errors


def main() -> None:
    if not RUNS_ROOT.exists():
        raise FileNotFoundError(f"Runs root not found: {RUNS_ROOT}")

    expected = {
        (lr, bs, seed)
        for lr in EXPECTED_LRS
        for bs in EXPECTED_BS
        for seed in EXPECTED_SEEDS
    }

    found_runs: dict[tuple[float, int, int], Path] = {}
    bad_name_dirs: list[Path] = []

    for p in sorted(RUNS_ROOT.iterdir()):
        if not p.is_dir():
            continue
        parsed = parse_run_name(p.name)
        if parsed is None:
            bad_name_dirs.append(p)
            continue
        key = (parsed["learning_rate"], parsed["batch_size"], parsed["seed"])
        found_runs[key] = p

    print("=== NAME CHECK ===")
    if bad_name_dirs:
        for p in bad_name_dirs:
            print(f"[bad-dir-name] {p}")
    else:
        print("all run directory names look valid")

    print("\n=== COVERAGE CHECK ===")
    missing = sorted(expected - set(found_runs.keys()))
    extra = sorted(set(found_runs.keys()) - expected)

    if missing:
        print("missing runs:")
        for lr, bs, seed in missing:
            print(f"  lr={lr} bs={bs} seed={seed}")
    else:
        print("no missing runs")

    if extra:
        print("extra runs:")
        for lr, bs, seed in extra:
            print(f"  lr={lr} bs={bs} seed={seed}")
    else:
        print("no extra runs")

    print("\n=== FILE / CONTENT VALIDATION ===")
    total_errors = 0
    checked = 0

    for key in sorted(found_runs.keys()):
        run_dir = found_runs[key]
        checked += 1
        errs = validate_one_run(run_dir)
        if errs:
            total_errors += len(errs)
            for e in errs:
                print(e)

    if total_errors == 0:
        print(f"all checked runs are valid ({checked} runs checked)")
    else:
        print(f"validation finished with {total_errors} errors across {checked} runs")


if __name__ == "__main__":
    main()
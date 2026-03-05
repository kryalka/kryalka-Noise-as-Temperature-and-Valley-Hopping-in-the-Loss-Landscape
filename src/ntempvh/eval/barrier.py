from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ntempvh.utils.io import load_yaml, save_json
import hashlib
from pathlib import Path

def _short_tag(p: str | Path) -> str:
    p = Path(p)
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:8]
    return f"{p.stem}__{h}"


def _parse_interp_csv(interp_csv: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Чтение CSV интерполяции (t, loss, acc); возвращает отсортированные по t массивы
    """
    arr = np.loadtxt(interp_csv, delimiter=",", skiprows=1)

    t = arr[:, 0].astype(np.float64)
    L = arr[:, 1].astype(np.float64)
    if arr.shape[1] >= 3:
        acc = arr[:, 2].astype(np.float64)
    else:
        acc = np.full_like(L, np.nan)

    order = np.argsort(t)
    return t[order], L[order], acc[order]


def _compute_deltaL(t: np.ndarray, L: np.ndarray, definition: str) -> tuple[float, float, float]:
    """
    Подсчет высоты барьера DeltaL по выбранному определению; возвращает (DeltaL, L0, L1)
    """
    L0 = float(L[0])
    L1 = float(L[-1])
    max_L = float(L.max())

    d = (definition or "").strip().lower()

    if d in ("max_loss_minus_endpoints", "max_minus_endpoints", "endpoints"):
        deltaL = max_L - max(L0, L1)
        return float(deltaL), L0, L1

    if d in ("max_minus_linear_baseline", "max_loss_minus_linear_baseline", "max_minus_linear", "linear"):
        baseline = (1.0 - t) * L0 + t * L1
        deltaL = float(np.max(L - baseline))
        return float(deltaL), L0, L1

    raise ValueError(f"Unknown barrier definition: {definition}")


def _safe_stem(p: Path) -> str:
    return str(p).replace("/", "__").replace("\\", "__").replace(":", "_")


def compute_barrier(interp_csv: str, barrier_cfg_path: str, out_path: str) -> Path:
    """
    Вычисление барьерв по CSV + YAML-конфигу 
    """
    cfg = load_yaml(barrier_cfg_path)
    bcfg = cfg.get("barrier", {}) if isinstance(cfg, dict) else {}

    definition = str(bcfg.get("definition", "max_minus_endpoints"))
    thresholds = [float(x) for x in bcfg.get("thresholds", [0.01, 0.05, 0.1])]

    t, L, acc = _parse_interp_csv(interp_csv)
    meta = None
    meta_path = Path(interp_csv).with_suffix(".meta.json")
    if meta_path.exists():
        try:
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = None
    # deltaL, L0, L1 = _compute_deltaL(t, L, definition=definition)
    # max_L = float(L.max())
    # peak_idx = int(L.argmax())
    # peak_t = float(t[peak_idx])
    deltaL, L0, L1 = _compute_deltaL(t, L, definition=definition)

    d = (definition or "").strip().lower()
    max_L = float(L.max())

    if d in ("max_minus_linear_baseline", "max_loss_minus_linear_baseline", "max_minus_linear", "linear"):
        baseline = (1.0 - t) * L0 + t * L1
        diff = L - baseline
        peak_idx = int(np.argmax(diff))
        peak_t = float(t[peak_idx])
        max_diff = float(np.max(diff))
    else:
        peak_idx = int(np.argmax(L))
        peak_t = float(t[peak_idx])
        max_diff = None

    eps = 1e-12
    deltaL_rel = float(deltaL / (0.5 * (L0 + L1) + eps))

    jumps = {str(th): bool(deltaL > th) for th in thresholds}

    out: dict[str, Any] = {
        "interp_csv": str(interp_csv),
        "definition": definition,
        "L0": float(L0),
        "L1": float(L1),
        "max_L": float(max_L),
        "peak_t": float(peak_t),
        "DeltaL": float(deltaL),
        "DeltaL_rel": float(deltaL_rel),
        "max_diff": max_diff,
        "thresholds": thresholds,
        "jumps": jumps,
    }

    if meta is not None:
        out["path"] = meta.get("path", None)
        out["evaluation"] = meta.get("evaluation", None)

    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    interp_p = Path(interp_csv)
    # tag = _safe_stem(interp_p.with_suffix(""))
    # json_path = out_dir / f"barrier__{tag}.json"
    tag = _short_tag(Path(interp_csv).with_suffix(""))
    json_path = out_dir / f"barrier__{tag}.json"
    save_json(json_path, out)

    # csv_path = out_dir / "barriers.csv"
    thr_sig = "_".join([str(x).replace(".", "p") for x in thresholds])
    def_sig = (definition or "").strip().lower().replace(" ", "_")
    csv_path = out_dir / f"barriers__{def_sig}__thr_{thr_sig}.csv"

    header = "interp_csv,definition,L0,L1,max_L,peak_t,DeltaL,DeltaL_rel," + ",".join(
        [f"jump_eps_{th}" for th in thresholds]
    )
    row = [
        str(interp_csv),
        definition,
        f"{L0:.10g}",
        f"{L1:.10g}",
        f"{max_L:.10g}",
        f"{peak_t:.10g}",
        f"{deltaL:.10g}",
        f"{deltaL_rel:.10g}",
    ] + [str(int(jumps[str(th)])) for th in thresholds]

    # if not csv_path.exists():
    #     csv_path.write_text(header + "\n", encoding="utf-8")
    # with open(csv_path, "a", encoding="utf-8") as f:
    #     f.write(",".join(row) + "\n")


    legacy_csv_path = out_dir / "barriers.csv"

    legacy_header = "interp_csv,definition,L0,L1,max_L,peak_t,DeltaL," + ",".join(
        [f"jump_eps_{th}" for th in thresholds]
    )

    legacy_row = [
        str(interp_csv),
        definition,
        f"{L0:.10g}",
        f"{L1:.10g}",
        f"{max_L:.10g}",
        f"{peak_t:.10g}",
        f"{deltaL:.10g}",
    ] + [str(int(jumps[str(th)])) for th in thresholds]

    if not legacy_csv_path.exists():
        legacy_csv_path.write_text(legacy_header + "\n", encoding="utf-8")
    with open(legacy_csv_path, "a", encoding="utf-8") as f:
        f.write(",".join(legacy_row) + "\n")
        
    return json_path
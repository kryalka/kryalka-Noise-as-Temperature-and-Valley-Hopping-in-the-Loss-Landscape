from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ntempvh.eval.barrier import compute_barrier


def _write_interp_csv(path: Path, t: np.ndarray, val_loss: np.ndarray, val_acc: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if val_acc is None:
        val_acc = np.full_like(val_loss, np.nan, dtype=float)

    arr = np.stack([t, val_loss, val_acc], axis=1)
    header = "t,val_loss,val_acc"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def _write_barrier_cfg(path: Path, definition: str, thresholds: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = (
        "barrier:\n"
        f"  definition: {definition}\n"
        "  thresholds:\n"
        + "".join([f"    - {x}\n" for x in thresholds])
    )
    path.write_text(txt, encoding="utf-8")


def test_compute_barrier_outputs_json_and_csv(tmp_path: Path) -> None:

    interp_csv = tmp_path / "some" / "nested" / "interp.csv"

    t = np.linspace(0.0, 1.0, 5)
    val_loss = np.array([1.0, 1.05, 1.2, 1.05, 1.0], dtype=float)
    val_acc = np.array([0.5, 0.55, 0.52, 0.56, 0.6], dtype=float)
    _write_interp_csv(interp_csv, t, val_loss, val_acc)

    cfg_path = tmp_path / "cfg" / "barrier.yaml"
    thresholds = [0.01, 0.05, 0.1]
    _write_barrier_cfg(cfg_path, definition="max_minus_endpoints", thresholds=thresholds)

    out_dir = tmp_path / "out"
    json_path = compute_barrier(str(interp_csv), str(cfg_path), str(out_dir))

    assert isinstance(json_path, Path)
    assert json_path.exists()
    assert json_path.parent.resolve() == out_dir.resolve()
    assert json_path.suffix == ".json"
    assert json_path.name.startswith("barrier__")

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["interp_csv"] == str(interp_csv)
    assert payload["definition"] == "max_minus_endpoints"
    assert abs(payload["L0"] - 1.0) < 1e-9
    assert abs(payload["L1"] - 1.0) < 1e-9
    assert abs(payload["max_L"] - 1.2) < 1e-9
    assert abs(payload["peak_t"] - 0.5) < 1e-9

    assert abs(payload["DeltaL"] - 0.2) < 1e-9

    jumps = payload["jumps"]
    for eps in thresholds:
        assert jumps[str(eps)] is True

    csv_path = out_dir / "barriers.csv"
    assert csv_path.exists()

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2  

    header = lines[0].split(",")
    assert header[:7] == ["interp_csv", "definition", "L0", "L1", "max_L", "peak_t", "DeltaL"]

    for eps in thresholds:
        assert f"jump_eps_{eps}" in header

    row = lines[-1].split(",")
    assert row[0] == str(interp_csv)
    assert row[1] == "max_minus_endpoints"


def test_compute_barrier_linear_baseline_definition(tmp_path: Path) -> None:
    interp_csv = tmp_path / "interp.csv"

    t = np.array([0.0, 0.5, 1.0], dtype=float)
    val_loss = np.array([1.0, 1.8, 2.0], dtype=float)
    _write_interp_csv(interp_csv, t, val_loss)

    cfg_path = tmp_path / "barrier.yaml"
    _write_barrier_cfg(cfg_path, definition="max_minus_linear_baseline", thresholds=[0.25, 0.35])

    out_dir = tmp_path / "out"
    json_path = compute_barrier(str(interp_csv), str(cfg_path), str(out_dir))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["definition"] == "max_minus_linear_baseline"

    assert abs(payload["DeltaL"] - 0.3) < 1e-9
    assert payload["jumps"]["0.25"] is True
    assert payload["jumps"]["0.35"] is False
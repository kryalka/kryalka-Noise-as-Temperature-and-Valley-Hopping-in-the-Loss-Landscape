from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ntempvh.eval.geometry import compute_geometry


@dataclass
class _DummyLoaders:
    train: DataLoader
    val: DataLoader
    bn: DataLoader


class _TinyNet(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


def _make_dummy_loaders(
    *,
    n_train: int = 32,
    n_val: int = 32,
    batch_size: int = 16,
    seed: int = 0,
) -> _DummyLoaders:
    g = torch.Generator().manual_seed(seed)

    x_train = torch.randn((n_train, 3, 32, 32), generator=g)
    y_train = torch.randint(0, 10, (n_train,), generator=g)
    x_val = torch.randn((n_val, 3, 32, 32), generator=g)
    y_val = torch.randint(0, 10, (n_val,), generator=g)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    bn_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return _DummyLoaders(train=train_loader, val=val_loader, bn=bn_loader)


def _write_geometry_cfg(path: Path) -> None:
    text = """\
        data_root: ./data
        geometry:
        alpha: 1e-3
        num_directions: 2
        eval_batch_size: 16
        num_eval_batches: 1
        """
    path.write_text(text, encoding="utf-8")


def test_compute_geometry_creates_json_and_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ntempvh.eval.geometry as geom_mod

    def fake_get_cifar10_loaders(*, root, batch_size, num_workers=2, pin_memory=True, val_batch_size=256):
        loaders = _make_dummy_loaders(n_train=32, n_val=32, batch_size=min(16, int(val_batch_size)), seed=123)
        return loaders

    def fake_make_model(name: str, num_classes: int):
        return _TinyNet(num_classes=num_classes)

    monkeypatch.setattr(geom_mod, "get_cifar10_loaders", fake_get_cifar10_loaders)
    monkeypatch.setattr(geom_mod, "make_model", fake_make_model)

    torch.manual_seed(0)
    model = _TinyNet(num_classes=10)
    ckpt = {
        "model": "resnet18",   
        "dataset": "cifar10",
        "seed": 0,
        "epoch": 1,
        "state_dict": model.state_dict(),
    }
    ckpt_path = tmp_path / "final.pt"
    torch.save(ckpt, ckpt_path)

    cfg_path = tmp_path / "geometry.yaml"
    _write_geometry_cfg(cfg_path)

    out_dir = tmp_path / "out"
    json_path = compute_geometry(str(ckpt_path), str(cfg_path), str(out_dir))

    assert json_path.exists()
    data = json_path.read_text(encoding="utf-8")
    assert '"kappa_tr"' in data
    assert '"epsilon"' in data
    assert '"base"' in data

    csv_path = out_dir / "geometries.csv"
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2 
    assert lines[0].startswith("ckpt,model,dataset,alpha,num_directions")


def test_compute_geometry_deterministic_under_fixed_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):

    import ntempvh.eval.geometry as geom_mod

    def fake_get_cifar10_loaders(*, root, batch_size, num_workers=2, pin_memory=True, val_batch_size=256):
        loaders = _make_dummy_loaders(n_train=32, n_val=32, batch_size=min(16, int(val_batch_size)), seed=999)
        return loaders

    def fake_make_model(name: str, num_classes: int):
        return _TinyNet(num_classes=num_classes)

    monkeypatch.setattr(geom_mod, "get_cifar10_loaders", fake_get_cifar10_loaders)
    monkeypatch.setattr(geom_mod, "make_model", fake_make_model)

    torch.manual_seed(0)
    model = _TinyNet(num_classes=10)
    ckpt = {"model": "resnet18", "dataset": "cifar10", "seed": 0, "epoch": 1, "state_dict": model.state_dict()}
    ckpt_path = tmp_path / "final.pt"
    torch.save(ckpt, ckpt_path)

    cfg_path = tmp_path / "geometry.yaml"
    _write_geometry_cfg(cfg_path)

    out_dir1 = tmp_path / "out1"
    out_dir2 = tmp_path / "out2"

    torch.manual_seed(12345)
    j1 = compute_geometry(str(ckpt_path), str(cfg_path), str(out_dir1))

    torch.manual_seed(12345)
    j2 = compute_geometry(str(ckpt_path), str(cfg_path), str(out_dir2))

    import json
    d1 = json.loads(j1.read_text(encoding="utf-8"))
    d2 = json.loads(j2.read_text(encoding="utf-8"))

    assert np.isfinite(d1["kappa_tr"])
    assert np.isfinite(d2["kappa_tr"])
    assert abs(float(d1["kappa_tr"]) - float(d2["kappa_tr"])) < 1e-10
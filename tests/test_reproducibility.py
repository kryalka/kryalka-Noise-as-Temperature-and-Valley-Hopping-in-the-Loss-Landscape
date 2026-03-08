from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ntempvh.train.trainer import train_one_run
from ntempvh.eval.interpolation import run_interpolation
from textwrap import dedent


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
        return self.fc(self.flatten(x))


def _make_dummy_loaders(
    *,
    n_train: int = 64,
    n_val: int = 64,
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    bn_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return _DummyLoaders(train=train_loader, val=val_loader, bn=bn_loader)

def _base_train_cfg(*, epochs: int = 2, batch_size: int = 16, lr: float = 0.1) -> dict:
    return {
        "dataset": "cifar10",
        "model": "resnet18",
        "data_root": "IGNORED_IN_TESTS",
        "training": {
            "optimizer": "sgd",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "nesterov": False,
            "scheduler": "none",
        },
        "logging": {
            "save_every_epochs": 0,  
            "save_final": True,
            "save_best": False,    
        },
    }


def _write_interp_cfg(path: Path, *, data_root: str, num_points: int, eval_batch_size: int) -> None:
    text = dedent(f"""\
    data_root: {data_root}

    path:
      type: linear
      num_points: {num_points}

    evaluation:
      model_mode: eval
      batch_size: {eval_batch_size}

    metrics:
      - val_loss
      - val_accuracy
    """)
    path.write_text(text, encoding="utf-8")

def _write_fake_ckpt(
    *,
    root: Path,
    run_name: str,
    epoch: int,
    payload: dict,
) -> Path:
    ckpt_dir = root / "outputs" / "runs_lr_bs_grid" / run_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path

def _state_dict_allclose(sd1: dict, sd2: dict, *, atol: float = 0.0, rtol: float = 0.0) -> bool:
    if sd1.keys() != sd2.keys():
        return False
    for k in sd1.keys():
        t1 = sd1[k]
        t2 = sd2[k]
        if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
            if t1.shape != t2.shape:
                return False
            if not torch.allclose(t1.cpu(), t2.cpu(), atol=atol, rtol=rtol):
                return False
        else:
            if t1 != t2:
                return False
    return True


def test_train_one_run_deterministic_given_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ntempvh.train.trainer as trainer_mod

    fixed_loaders = _make_dummy_loaders(n_train=64, n_val=64, batch_size=16, seed=999)

    def fake_get_cifar10_loaders(*, root, batch_size, num_workers=2, pin_memory=True, val_batch_size=256):
        return fixed_loaders

    def fake_make_model(name: str, num_classes: int):
        return _TinyNet(num_classes=num_classes)

    monkeypatch.setattr(trainer_mod, "get_cifar10_loaders", fake_get_cifar10_loaders)
    monkeypatch.setattr(trainer_mod, "make_model", fake_make_model)
    monkeypatch.setattr(trainer_mod, "get_device", lambda: torch.device("cpu"))

    cfg = _base_train_cfg(epochs=2, batch_size=16, lr=0.05)

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    seed = 12345
    ckpt1 = train_one_run(cfg, seed=seed, out_dir=str(out1))
    ckpt2 = train_one_run(cfg, seed=seed, out_dir=str(out2))

    assert ckpt1.exists()
    assert ckpt2.exists()

    A = torch.load(ckpt1, map_location="cpu")
    B = torch.load(ckpt2, map_location="cpu")

    assert A["model"] == B["model"]
    assert A["dataset"] == B["dataset"]
    assert int(A["epoch"]) == int(B["epoch"])

    assert _state_dict_allclose(A["state_dict"], B["state_dict"], atol=0.0, rtol=0.0)


def test_interpolation_deterministic_given_fixed_ckpts_and_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ntempvh.eval.interpolation as interp_mod

    fixed_loaders = _make_dummy_loaders(n_train=32, n_val=32, batch_size=16, seed=2024)

    def fake_get_cifar10_loaders(*, root, batch_size, num_workers=2, pin_memory=True, val_batch_size=256):
        return fixed_loaders

    def fake_make_model(name: str, num_classes: int):
        return _TinyNet(num_classes=num_classes)

    monkeypatch.setattr(interp_mod, "get_cifar10_loaders", fake_get_cifar10_loaders)
    monkeypatch.setattr(interp_mod, "make_model", fake_make_model)
    monkeypatch.setattr(interp_mod, "get_device", lambda: torch.device("cpu"))

    torch.manual_seed(0)
    modelA = _TinyNet(num_classes=10)
    torch.manual_seed(1)
    modelB = _TinyNet(num_classes=10)

    run_name = "cifar10_resnet18_seed1__optsgd_lr0.1_bs128_wd0.0005_mom0.9_schnone__dummy"

    ckptA = {
        "model": "resnet18",
        "dataset": "cifar10",
        "seed": 1,
        "epoch": 1,
        "state_dict": modelA.state_dict(),
    }
    ckptB = {
        "model": "resnet18",
        "dataset": "cifar10",
        "seed": 1,
        "epoch": 10,
        "state_dict": modelB.state_dict(),
    }

    ckptA_path = _write_fake_ckpt(
        root=tmp_path,
        run_name=run_name,
        epoch=1,
        payload=ckptA,
    )
    ckptB_path = _write_fake_ckpt(
        root=tmp_path,
        run_name=run_name,
        epoch=10,
        payload=ckptB,
    )

    cfg_path = tmp_path / "interpolation.yaml"
    _write_interp_cfg(cfg_path, data_root="IGNORED", num_points=7, eval_batch_size=16)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    torch.manual_seed(123)
    csv1 = run_interpolation(str(ckptA_path), str(ckptB_path), str(cfg_path), str(out1))

    torch.manual_seed(123)
    csv2 = run_interpolation(str(ckptA_path), str(ckptB_path), str(cfg_path), str(out2))

    a1 = np.loadtxt(csv1, delimiter=",", skiprows=1)
    a2 = np.loadtxt(csv2, delimiter=",", skiprows=1)

    assert a1.shape == a2.shape
    assert np.allclose(a1, a2, atol=0.0, rtol=0.0)
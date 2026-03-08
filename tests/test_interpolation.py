from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from textwrap import dedent
from ntempvh.eval.interpolation import run_interpolation


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

@torch.no_grad()
def _eval_like_module(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        n += int(x.size(0))
    return loss_sum / max(1, n), correct / max(1, n)


def test_run_interpolation_uses_config_and_writes_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ntempvh.eval.interpolation as interp_mod

    expected_root = str(tmp_path / "my_data_root")
    expected_eval_bs = 16
    num_points = 5

    seen = {"root": None, "val_batch_size": None}

    def fake_get_cifar10_loaders(*, root, batch_size, num_workers=2, pin_memory=True, val_batch_size=256):
        seen["root"] = root
        seen["val_batch_size"] = int(val_batch_size)
        loaders = _make_dummy_loaders(n_train=32, n_val=32, batch_size=min(expected_eval_bs, int(val_batch_size)), seed=123)
        return loaders

    def fake_make_model(name: str, num_classes: int):
        return _TinyNet(num_classes=num_classes)

    monkeypatch.setattr(interp_mod, "get_cifar10_loaders", fake_get_cifar10_loaders)
    monkeypatch.setattr(interp_mod, "make_model", fake_make_model)

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
    _write_interp_cfg(cfg_path, data_root=expected_root, num_points=num_points, eval_batch_size=expected_eval_bs)

    out_dir = tmp_path / "out"
    out_csv = run_interpolation(str(ckptA_path), str(ckptB_path), str(cfg_path), str(out_dir))

    assert seen["root"] == expected_root
    assert seen["val_batch_size"] == expected_eval_bs

    assert out_csv.exists()
    txt = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(txt) == 1 + num_points  
    assert txt[0].strip() == "t,val_loss,val_acc"

    arr = np.loadtxt(out_csv, delimiter=",", skiprows=1)
    assert arr.shape == (num_points, 3)

    t = arr[:, 0]
    assert np.allclose(t, np.linspace(0.0, 1.0, num_points), atol=0.0, rtol=0.0)

    assert np.all(np.isfinite(arr[:, 1]))
    assert np.all(np.isfinite(arr[:, 2]))


def test_run_interpolation_endpoints_match_manual_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    import ntempvh.eval.interpolation as interp_mod
    device = torch.device("cpu")

    expected_root = str(tmp_path / "data_root")
    eval_bs = 16
    num_points = 3  
    fixed_loaders = _make_dummy_loaders(n_train=32, n_val=32, batch_size=eval_bs, seed=777)

    def fake_get_cifar10_loaders(*, root, batch_size, num_workers=2, pin_memory=True, val_batch_size=256):
        return fixed_loaders

    def fake_make_model(name: str, num_classes: int):
        return _TinyNet(num_classes=num_classes)

    monkeypatch.setattr(interp_mod, "get_cifar10_loaders", fake_get_cifar10_loaders)
    monkeypatch.setattr(interp_mod, "make_model", fake_make_model)
    monkeypatch.setattr(interp_mod, "get_device", lambda: device)

    torch.manual_seed(123)
    modelA = _TinyNet(num_classes=10)
    torch.manual_seed(456)
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
    _write_interp_cfg(cfg_path, data_root=expected_root, num_points=num_points, eval_batch_size=eval_bs)

    out_dir = tmp_path / "out"
    out_csv = run_interpolation(str(ckptA_path), str(ckptB_path), str(cfg_path), str(out_dir))
    arr = np.loadtxt(out_csv, delimiter=",", skiprows=1)

    mA = _TinyNet(num_classes=10)
    mA.load_state_dict(modelA.state_dict(), strict=True)
    lossA, accA = _eval_like_module(mA, fixed_loaders.val, device)

    mB = _TinyNet(num_classes=10)
    mB.load_state_dict(modelB.state_dict(), strict=True)
    lossB, accB = _eval_like_module(mB, fixed_loaders.val, device)

    assert abs(arr[0, 1] - lossA) < 1e-10
    assert abs(arr[0, 2] - accA) < 1e-10

    assert abs(arr[-1, 1] - lossB) < 1e-10
    assert abs(arr[-1, 2] - accB) < 1e-10
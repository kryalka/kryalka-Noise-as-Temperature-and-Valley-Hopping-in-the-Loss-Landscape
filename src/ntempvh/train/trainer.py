from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from ntempvh.data.cifar import get_cifar10_loaders
from ntempvh.models.resnet_cifar import make_model
from ntempvh.train.optim import make_optimizer
from ntempvh.train.schedules import make_scheduler, step_scheduler
from ntempvh.utils.device import get_device
from ntempvh.utils.io import ensure_dir, save_json
from ntempvh.utils.logging import RunLogger
from ntempvh.utils.seed import set_seed

import inspect

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    loss_sum = 0.0
    correct = 0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss_sum += float(criterion(logits, y).item())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        n += int(x.size(0))

    return {
        "val_loss": loss_sum / max(1, n),
        "val_acc": correct / max(1, n),
    }

def _call_get_cifar10_loaders_safe(**kwargs):
    sig = inspect.signature(get_cifar10_loaders)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return get_cifar10_loaders(**filtered)


def _save_checkpoint(
    ckpt_dir: Path,
    tag: str,
    *,
    model_name: str,
    dataset_name: str,
    seed: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    ckpt = {
        "model": model_name,
        "dataset": dataset_name,
        "seed": int(seed),
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    if extra:
        ckpt.update(extra)

    path = ckpt_dir / f"{tag}.pt"
    torch.save(ckpt, path)
    return path


def train_one_run(config: Dict[str, Any], seed: int, out_dir: str) -> Path:
    device = get_device()
    set_seed(seed)

    out_path = ensure_dir(out_dir)
    ckpt_dir = ensure_dir(out_path / "checkpoints")
    logger = RunLogger(out_path)

    dataset_name = str(config["dataset"]).lower()
    model_name = str(config["model"]).lower()
    train_cfg: Dict[str, Any] = dict(config["training"])

    log_cfg: Dict[str, Any] = dict(config.get("logging", {}))
    save_every_epochs = int(log_cfg.get("save_every_epochs", 0) or 0)
    save_final = bool(log_cfg.get("save_final", True))
    save_best = bool(log_cfg.get("save_best", True))

    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])

    data_root = str(config.get("data_root", "./data"))

    data_cfg: Dict[str, Any] = dict(config.get("data", {}))
    val_size = int(data_cfg.get("val_size", 5000))
    split_seed = int(data_cfg.get("split_seed", 0))
    num_workers = int(data_cfg.get("num_workers", 0))  
    pin_memory = bool(data_cfg.get("pin_memory", device.type in ("cuda",)))  

    loaders = _call_get_cifar10_loaders_safe(
        root=data_root,
        batch_size=batch_size,
        val_size=val_size,
        split_seed=split_seed,
        shuffle_seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = make_model(model_name, num_classes=10).to(device)
    optimizer = make_optimizer(train_cfg, model)
    scheduler = make_scheduler(train_cfg, optimizer)
    criterion = nn.CrossEntropyLoss()

    save_json(out_path / "run_config.json", {"seed": int(seed), "device": str(device), **config})

    best_val_loss = float("inf")
    best_ckpt_path: Optional[Path] = None
    best_epoch: Optional[int] = None
    last_val: Optional[Dict[str, float]] = None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(loaders.train, desc=f"epoch {ep}/{epochs}", leave=False)

        running_loss = 0.0
        seen = 0
        correct_train = 0

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y) 
            pred = logits.argmax(dim=1)
            correct_train += int((pred == y).sum().item())
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)

            pbar.set_postfix(
                train_loss=running_loss / max(1, seen),
                lr=float(optimizer.param_groups[0]["lr"]),
            )

        step_scheduler(scheduler)

        val = evaluate(model, loaders.val, device)
        last_val = val
        train_loss_ep = running_loss / max(1, seen)
        train_acc_ep = correct_train / max(1, seen)

        logger.log({
            "epoch": int(ep),
            "train_loss": float(train_loss_ep),
            "train_acc": float(train_acc_ep),
            "val_loss": float(val["val_loss"]),
            "val_acc": float(val["val_acc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds_elapsed": float(time.time() - t0),
        })

        print(f"[ep {ep:03d}] val_loss={val['val_loss']:.4f} val_acc={val['val_acc']:.4f} train_acc={train_acc_ep:.4f}")

        if save_best and val["val_loss"] < best_val_loss:
            best_val_loss = float(val["val_loss"])
            best_epoch = ep
            best_ckpt_path = _save_checkpoint(
                ckpt_dir,
                "best",
                model_name=model_name,
                dataset_name=dataset_name,
                seed=seed,
                epoch=ep,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                extra={"best_val_loss": best_val_loss},
            )

        if save_every_epochs > 0 and (ep % save_every_epochs == 0):
            _save_checkpoint(
                ckpt_dir,
                f"epoch_{ep:03d}",
                model_name=model_name,
                dataset_name=dataset_name,
                seed=seed,
                epoch=ep,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    final_ckpt_path: Path
    if save_final:
        final_ckpt_path = _save_checkpoint(
            ckpt_dir,
            "final",
            model_name=model_name,
            dataset_name=dataset_name,
            seed=seed,
            epoch=epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    else:
        final_ckpt_path = ckpt_dir / "final.pt"
        torch.save(
            {
                "model": model_name,
                "dataset": dataset_name,
                "seed": int(seed),
                "epoch": int(epochs),
                "state_dict": model.state_dict(),
            },
            final_ckpt_path,
        )

    save_json(out_path / "summary.json", {
        "seed": int(seed),
        "epochs": int(epochs),
        "seconds_total": float(time.time() - t0),
        "final_checkpoint": str(final_ckpt_path),
        "best_checkpoint": str(best_ckpt_path) if best_ckpt_path is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss < float("inf") else None,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "final_val_loss": float(last_val["val_loss"]) if last_val is not None else None,
        "final_val_acc": float(last_val["val_acc"]) if last_val is not None else None,
        "save_every_epochs": int(save_every_epochs),
        "data": {
            "data_root": data_root,
            "val_size": val_size,
            "split_seed": split_seed,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
    })

    return final_ckpt_path
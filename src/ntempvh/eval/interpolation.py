from __future__ import annotations

from pathlib import Path
import inspect

import numpy as np
import torch
import torch.nn as nn

from ntempvh.data.cifar import get_cifar10_loaders, get_cifar10_test_loader
from ntempvh.models.resnet_cifar import make_model
from ntempvh.utils.device import get_device
from ntempvh.utils.io import ensure_dir, load_yaml


def _lerp_state_dict(sd_a: dict, sd_b: dict, t: float) -> dict:
    """
    Формирование промежуточной модели для анализа связности мод (mode connectivity) по линейной интерполяции весов
    """
    out = {}
    for k in sd_a.keys():
        a = sd_a[k]
        b = sd_b[k]

        is_bn_buf = ("running_mean" in k) or ("running_var" in k) or ("num_batches_tracked" in k)
        if is_bn_buf or (hasattr(a, "is_floating_point") and (not a.is_floating_point())):
            out[k] = a.clone()
            continue

        out[k] = (1.0 - t) * a + t * b
    return out

def _interp_state_dicts_piecewise(
    sds: list[dict],
    t: float,
) -> dict:
    """
    Piecewise-linear interpolation through a sequence of checkpoints:
    sd[0] -> sd[1] -> ... -> sd[K-1].
    t in [0,1] is mapped to a segment and local tau in [0,1].
    """
    k = len(sds) - 1
    u = float(np.clip(t, 0.0, 1.0)) * k
    i = int(min(k - 1, max(0, np.floor(u))))
    tau = u - i
    return _lerp_state_dict(sds[i], sds[i + 1], float(tau))

@torch.no_grad()
def _eval(model: nn.Module, loader, device) -> tuple[float, float]:
    """
    Подсчет целевой метрики качества на сплите (loss/accuracy) для сравнения точек вдоль интерполяционного пути
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        loss_sum += float(criterion(logits, y).item())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        n += int(x.size(0))

    if n <= 0:
        return float("nan"), float("nan")
    return loss_sum / n, correct / n


def _call_get_cifar10_loaders_safe(**kwargs):
    sig = inspect.signature(get_cifar10_loaders)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return get_cifar10_loaders(**filtered)

@torch.no_grad()
def recalibrate_bn(
    model: nn.Module,
    train_loader,
    device: torch.device,
    *,
    num_batches: int = 50,
) -> None:
    """
    Обновление running_mean / running_var у BatchNorm слоёв под текущие веса модели
    Модель в eval, но BN-слои в train (обновятся running stats без dropout)
    """
    bn_layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bn_layers or num_batches <= 0:
        return

    model.eval()
    for m in bn_layers:
        m.train()

    batches = 0
    for x, _ in train_loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        batches += 1
        if batches >= num_batches:
            break

    model.eval()

def run_interpolation(ckpt_a: str, ckpt_b: str, config_path: str, out_dir: str) -> Path:
    """
    Вывод кривой барьера (loss/acc) вдоль пути между двумя чекпоинтами 
    """
    cfg = load_yaml(config_path)

    path_cfg = cfg.get("path", {}) if isinstance(cfg, dict) else {}
    if path_cfg is None:
        path_cfg = {}

    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    if data_cfg is None:
        data_cfg = {}

    num_points = int(path_cfg.get("num_points", cfg.get("num_points", 41)))
    bn_batches = int(path_cfg.get("bn_recalib_batches", cfg.get("bn_recalib_batches", 0)))

    path_type = str(path_cfg.get("type", "linear")).strip().lower()
    pivot_paths = path_cfg.get("pivots", []) or []
    pivot_paths = [str(p) for p in pivot_paths]

    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
    if eval_cfg is None:
        eval_cfg = {}

    data_root = str(cfg.get("data_root", "./data"))
    eval_batch_size = int(eval_cfg.get("batch_size", cfg.get("batch_size", 256)))
    eval_split = str(eval_cfg.get("split", cfg.get("split", "val"))).strip().lower()

    device = get_device()
    out_path = ensure_dir(out_dir)

    A = torch.load(ckpt_a, map_location="cpu")
    B = torch.load(ckpt_b, map_location="cpu")

    model_name = str(A["model"]).lower()
    dataset_name = str(A.get("dataset", "")).lower() 

    sd_list = [A["state_dict"]]

    for p in pivot_paths:
        P = torch.load(p, map_location="cpu")
        sd_list.append(P["state_dict"])

    sd_list.append(B["state_dict"])

    # sdA = A["state_dict"]
    # sdB = B["state_dict"]

    # if eval_split == "val":
    #     loaders = _call_get_cifar10_loaders_safe(
    #         root=data_root,
    #         batch_size=128,  
    #         val_batch_size=eval_batch_size,
    #         val_size=int(eval_cfg.get("val_size", 5000)),
    #         split_seed=int(eval_cfg.get("split_seed", 0)),
    #         num_workers=int(data_cfg.get("num_workers", 0)),
    #         pin_memory=bool(data_cfg.get("pin_memory", True)),
    #     )
    #     eval_loader = loaders.val
    #     train_loader = loaders.train
        

    # elif eval_split == "test":
    #     eval_loader = get_cifar10_test_loader(
    #         root=data_root,
    #         batch_size=eval_batch_size,
    #         num_workers=int(data_cfg.get("num_workers", 0)),
    #         pin_memory=bool(data_cfg.get("pin_memory", True)),
    #     )

    loaders = _call_get_cifar10_loaders_safe(
        root=data_root,
        batch_size=128,
        val_batch_size=eval_batch_size,
        val_size=int(eval_cfg.get("val_size", 5000)),
        split_seed=int(eval_cfg.get("split_seed", 0)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    bn_loader = loaders.bn  
    if bn_loader is None:
        bn_loader = getattr(loaders, "train", None) or getattr(loaders, "val", None)

    if eval_split == "val":
        eval_loader = loaders.val
    elif eval_split == "test":
        eval_loader = get_cifar10_test_loader(
            root=data_root,
            batch_size=eval_batch_size,
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=bool(data_cfg.get("pin_memory", True)),
        )

    model = make_model(model_name, num_classes=10).to(device)

    ts = np.linspace(0.0, 1.0, num_points)
    rows = []
    for t in ts:
        if path_type == "linear":
            sd = _lerp_state_dict(sd_list[0], sd_list[-1], float(t))
        elif path_type in ("polyline", "piecewise", "piecewise_linear"):
            sd = _interp_state_dicts_piecewise(sd_list, float(t))
        else:
            raise ValueError(f"Unknown path.type: {path_type}")

        model.load_state_dict(sd, strict=True)

        recalibrate_bn(model, bn_loader, device, num_batches=bn_batches)

        val_loss, val_acc = _eval(model, eval_loader, device)
        rows.append([float(t), float(val_loss), float(val_acc)])

    arr = np.array(rows, dtype=np.float64)
    def _safe_stem(p: str) -> str:
        return str(Path(p).with_suffix("")).replace("/", "__").replace("\\", "__").replace(":", "_")

    tag = f"interp__{path_type}__{_safe_stem(ckpt_a)}__{_safe_stem(ckpt_b)}.csv"
    out_file = Path(out_path) / tag

    np.savetxt(out_file, arr, delimiter=",", header="t,val_loss,val_acc", comments="")

    return out_file
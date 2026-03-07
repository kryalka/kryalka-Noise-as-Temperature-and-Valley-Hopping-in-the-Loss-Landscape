from __future__ import annotations

from pathlib import Path
import inspect

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from ntempvh.utils.seed import set_seed
from ntempvh.data.cifar import get_cifar10_loaders, get_cifar10_test_loader
from ntempvh.models.resnet_cifar import make_model
from ntempvh.utils.device import get_device
from ntempvh.utils.io import ensure_dir, load_yaml, save_json
from ntempvh.eval.bn import recalibrate_bn
import re
from pathlib import Path

RUN_RE = re.compile(
    r"""
    seed(?P<seed>\d+)
    __opt(?P<optimizer>[^_]+)
    _lr(?P<lr>[^_]+)
    _bs(?P<bs>[^_]+)
    _wd(?P<wd>[^_]+)
    _mom(?P<mom>[^_]+)
    _sch(?P<scheduler>[^_]+)
    __(?P<hash>[a-f0-9]+)
    """,
    re.VERBOSE,
)

EPOCH_RE = re.compile(r"^epoch_(\d+)\.pt$")


def _parse_ckpt_path(ckpt_path: str | Path) -> dict:
    p = Path(ckpt_path)
    run_name = p.parent.parent.name

    m = RUN_RE.search(run_name)
    em = EPOCH_RE.match(p.name)

    g = m.groupdict()
    return {
        "run_name": run_name,
        "seed": int(g["seed"]),
        "learning_rate": float(g["lr"]),
        "batch_size": int(g["bs"]),
        "optimizer": g["optimizer"],
        "weight_decay": float(g["wd"]),
        "momentum": float(g["mom"]),
        "scheduler": g["scheduler"],
        "epoch": int(em.group(1)),
    }


def _lr_to_str(x: float) -> str:
    return f"{x:g}"


def _pair_tag_from_ckpts(ckpt_a: str | Path, ckpt_b: str | Path) -> str:
    a = _parse_ckpt_path(ckpt_a)
    b = _parse_ckpt_path(ckpt_b)

    if a["run_name"] != b["run_name"]:
        raise ValueError(
            f"Interpolation is expected within a single run, got:\\n"
            f"  A: {ckpt_a}\\n"
            f"  B: {ckpt_b}"
        )

    return (
        f"lr{_lr_to_str(a['learning_rate'])}"
        f"__bs{a['batch_size']}"
        f"__seed{a['seed']}"
        f"__e{a['epoch']:03d}_e{b['epoch']:03d}"
    )


def _lerp_state_dict(sd_a: dict, sd_b: dict, t: float) -> dict:
    """
    Формирование промежуточной модели для анализа mode connectivity по линейной интерполяции весов
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

@torch.no_grad()
def _eval_endpoint_state_dict(
    *,
    model_name: str,
    state_dict: dict,
    bn_loader,
    eval_loader,
    device: torch.device,
    bn_batches: int,
) -> tuple[float, float]:
    """
    Оценка endpoint-чекпоинта на свежей модели:
    создаём новую модель, грузим state_dict, делаем BN recalibration, считаем метрики.
    """
    endpoint_model = make_model(model_name, num_classes=10).to(device)
    endpoint_model.load_state_dict(state_dict, strict=True)
    recalibrate_bn(endpoint_model, bn_loader, device, num_batches=bn_batches, reset_stats=True)
    return _eval(endpoint_model, eval_loader, device)


def _call_get_cifar10_loaders_safe(**kwargs):
    sig = inspect.signature(get_cifar10_loaders)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return get_cifar10_loaders(**filtered)

def _validate_ckpt_pair(A: dict, B: dict) -> None:
    model_a = str(A.get("model", "")).lower()
    model_b = str(B.get("model", "")).lower()
    
    ds_a = str(A.get("dataset", "")).lower()
    ds_b = str(B.get("dataset", "")).lower()
    if ds_a != ds_b:
        raise ValueError(f"Checkpoint dataset mismatch: {ds_a} vs {ds_b}")

    sda = A["state_dict"]
    sdb = B["state_dict"]


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
    seed = int(A.get("seed", 0))
    set_seed(seed)

    _validate_ckpt_pair(A, B)

    model_name = str(A["model"]).lower()
    dataset_name = str(A.get("dataset", "")).lower() 

    sd_list = [A["state_dict"]]

    for p in pivot_paths:
        P = torch.load(p, map_location="cpu")
        sd_list.append(P["state_dict"])

    sd_list.append(B["state_dict"])

    bn_batch_size = int(eval_cfg.get("bn_batch_size", eval_batch_size))

    loaders = _call_get_cifar10_loaders_safe(
        root=data_root,
        batch_size=128,
        val_batch_size=eval_batch_size,
        val_size=int(eval_cfg.get("val_size", 5000)),
        bn_batch_size=bn_batch_size,
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

    endpoint_A_loss, endpoint_A_acc = _eval_endpoint_state_dict(
        model_name=model_name,
        state_dict=sd_list[0],
        bn_loader=bn_loader,
        eval_loader=eval_loader,
        device=device,
        bn_batches=bn_batches,
    )

    endpoint_B_loss, endpoint_B_acc = _eval_endpoint_state_dict(
        model_name=model_name,
        state_dict=sd_list[-1],
        bn_loader=bn_loader,
        eval_loader=eval_loader,
        device=device,
        bn_batches=bn_batches,
    )

    ts = np.linspace(0.0, 1.0, num_points)
    rows = []
    pbar = tqdm(ts, desc="interpolation points", total=len(ts))
    print(
        f"[interpolation] path_type={path_type}, num_points={num_points}, "
        f"split={eval_split}, bn_batches={bn_batches}"
    )
    for t in pbar:
        if path_type == "linear":
            sd = _lerp_state_dict(sd_list[0], sd_list[-1], float(t))
        elif path_type in ("polyline", "piecewise", "piecewise_linear"):
            sd = _interp_state_dicts_piecewise(sd_list, float(t))
        else:
            raise ValueError(f"Unknown path.type: {path_type}")

        model.load_state_dict(sd, strict=True)

        recalibrate_bn(model, bn_loader, device, num_batches=bn_batches, reset_stats=True)

        val_loss, val_acc = _eval(model, eval_loader, device)
        rows.append([float(t), float(val_loss), float(val_acc)])

        pbar.set_postfix(
            t=f"{float(t):.2f}",
            val_loss=f"{float(val_loss):.4f}",
            val_acc=f"{float(val_acc):.4f}",
        )

    arr = np.array(rows, dtype=np.float64)
    arr[0, 1] = float(endpoint_A_loss)
    arr[0, 2] = float(endpoint_A_acc)
    arr[-1, 1] = float(endpoint_B_loss)
    arr[-1, 2] = float(endpoint_B_acc)
    
    pair_tag = _pair_tag_from_ckpts(ckpt_a, ckpt_b)
    out_file = Path(out_path) / f"interp__{pair_tag}.csv"

    np.savetxt(out_file, arr, delimiter=",", header="t,val_loss,val_acc", comments="")


    meta = {
        "ckptA": str(ckpt_a),
        "ckptB": str(ckpt_b),
        "model": model_name,
        "dataset": dataset_name,
        "data_root": data_root,
        "path": {
            "type": path_type,
            "num_points": int(num_points),
            "bn_recalib_batches": int(bn_batches),
            "pivots": pivot_paths,
        },
        "evaluation": {
            "split": eval_split,
            "batch_size": int(eval_batch_size),
            "val_size": int(eval_cfg.get("val_size", 5000)),
            "split_seed": int(eval_cfg.get("split_seed", 0)),
        },
        "endpoint_eval": {
            "A": {"loss": float(endpoint_A_loss), "acc": float(endpoint_A_acc)},
            "B": {"loss": float(endpoint_B_loss), "acc": float(endpoint_B_acc)},
        },
    }

    for tag, ck in (("A", A), ("B", B)):
        if isinstance(ck, dict):
            if "epoch" in ck:
                meta[f"epoch_{tag}"] = int(ck["epoch"])
            if "seed" in ck:
                meta[f"seed_{tag}"] = int(ck["seed"])

    meta_path = Path(out_file).with_suffix(".meta.json")
    save_json(meta_path, meta)


    return out_file
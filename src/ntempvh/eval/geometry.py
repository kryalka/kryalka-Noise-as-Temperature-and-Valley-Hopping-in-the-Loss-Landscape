from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm
import time

import numpy as np
import torch
from ntempvh.utils.seed import set_seed
from ntempvh.data.cifar import get_cifar10_loaders
from ntempvh.models.resnet_cifar import make_model
from ntempvh.utils.device import get_device
from ntempvh.utils.io import ensure_dir, load_yaml, save_json
from ntempvh.eval.metrics import eval_classification, params_to_vector, vector_to_params
from ntempvh.eval.bn import recalibrate_bn
import inspect
import hashlib
from pathlib import Path

def _short_tag(p: str | Path) -> str:
    p = Path(p)
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:8]
    return f"{p.stem}__{h}"

def _call_get_cifar10_loaders_safe(**kwargs):
    sig = inspect.signature(get_cifar10_loaders)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return get_cifar10_loaders(**filtered)


def _safe_stem(p: Path) -> str:
    return str(p).replace("/", "__").replace("\\", "__").replace(":", "_")


def _sample_unit_directions(
    d: int, m: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Сэмплирование m случайных единичных направлений в R^d (по норме L2)
    """
    z = torch.randn((m, d), device=device, dtype=dtype)
    z_norm = torch.norm(z, dim=1, keepdim=True).clamp_min(1e-12)
    return z / z_norm

def _save_failure_json(
    *,
    out_dir: str | Path,
    ckpt_path: str,
    model_name: str,
    dataset_name: str,
    device: torch.device,
    alpha: float,
    m: int,
    eval_batch_size: int,
    num_eval_batches: int | None,
    bn_batches: int,
    raw_base: dict | None,
    bn_base: dict | None,
    reason: str,
    extra: dict | None = None,
) -> Path:
    out_dir = ensure_dir(out_dir)
    tag = _short_tag(ckpt_path)
    fail_path = Path(out_dir) / f"geometry_failed__{tag}.json"

    payload: dict[str, Any] = {
        "status": "failed",
        "reason": reason,
        "ckpt": str(ckpt_path),
        "dataset": dataset_name,
        "model": model_name,
        "device": str(device),
        "alpha": alpha,
        "num_directions": m,
        "eval_batch_size": eval_batch_size,
        "num_eval_batches": num_eval_batches,
        "bn_recalib_batches": bn_batches,
        "raw_base": raw_base,
        "bn_base": bn_base,
    }
    if extra:
        payload["extra"] = extra

    save_json(fail_path, payload)
    return fail_path


@torch.no_grad()
def compute_geometry(ckpt_path: str, geometry_cfg_path: str, out_path: str) -> Path:
    """
    Оценка локальной кривизны около чекпоинта по случайным направлениям
    """
    cfg = load_yaml(geometry_cfg_path)
    gcfg = cfg.get("geometry", {}) if isinstance(cfg, dict) else {}
    if gcfg is None:
        gcfg = {}

    alpha = float(gcfg.get("alpha", cfg.get("alpha", 1e-3)))
    m = int(gcfg.get("num_directions", cfg.get("num_directions", 10)))
    eval_batch_size = int(gcfg.get("eval_batch_size", cfg.get("eval_batch_size", 256)))
    bn_batches = int(gcfg.get("bn_recalib_batches", cfg.get("bn_recalib_batches", 0)))

    num_eval_batches = gcfg.get("num_eval_batches", cfg.get("num_eval_batches", None))
    num_eval_batches = int(num_eval_batches) if num_eval_batches is not None else None

    device = get_device()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    seed = int(ckpt.get("seed", 0))      
    set_seed(seed) 
    
    model_name = str(ckpt["model"]).lower()
    dataset_name = str(ckpt.get("dataset", "")).lower()

    data_root = str(cfg.get("data_root", "./data"))

    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
    if eval_cfg is None:
        eval_cfg = {}

    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    if data_cfg is None:
        data_cfg = {}


    loaders = _call_get_cifar10_loaders_safe(
        root=data_root,
        batch_size=eval_batch_size,
        val_batch_size=eval_batch_size,
        val_size=int(eval_cfg.get("val_size", 5000)),
        split_seed=int(eval_cfg.get("split_seed", 0)),
        shuffle_seed=int(eval_cfg.get("split_seed", 0)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    val_loader = loaders.val
    bn_loader = loaders.bn

    model = make_model(model_name, num_classes=10).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    raw_base = eval_classification(model, val_loader, device, max_batches=num_eval_batches)
    # print("raw_base:", raw_base)
    recalibrate_bn(model, bn_loader, device, num_batches=bn_batches, reset_stats=False)
    base = eval_classification(model, val_loader, device, max_batches=num_eval_batches)
    # print("bn_base:", base)

    base_loss = float(base["loss"])
    base_acc = float(base["acc"])

    if not np.isfinite(base_loss) or not np.isfinite(base_acc):
        fail_path = _save_failure_json(
            out_dir=out_path,
            ckpt_path=ckpt_path,
            model_name=model_name,
            dataset_name=dataset_name,
            device=device,
            alpha=alpha,
            m=m,
            eval_batch_size=eval_batch_size,
            num_eval_batches=num_eval_batches,
            bn_batches=bn_batches,
            raw_base=raw_base,
            bn_base=base,
            reason="non_finite_bn_base",
        )

    if base_acc < 0.2 or base_loss > 2.5:
        fail_path = _save_failure_json(
            out_dir=out_path,
            ckpt_path=ckpt_path,
            model_name=model_name,
            dataset_name=dataset_name,
            device=device,
            alpha=alpha,
            m=m,
            eval_batch_size=eval_batch_size,
            num_eval_batches=num_eval_batches,
            bn_batches=bn_batches,
            raw_base=raw_base,
            bn_base=base,
            reason="unstable_bn_recalibration",
            extra={
                "threshold_acc_min": 0.2,
                "threshold_loss_max": 2.5,
            },
        )

    L0 = base_loss

    theta0 = params_to_vector(model).detach().to(device)
    dtype = theta0.dtype

    theta_norm = float(torch.norm(theta0).item())
    eps = float(alpha * theta_norm)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"Bad epsilon: eps={eps}, alpha={alpha}, ||theta||={theta_norm}")

    d = int(theta0.numel())
    U = _sample_unit_directions(d, m, device=device, dtype=dtype)

    per_dir: list[float] = []
    start_total = time.time()

    pbar = tqdm(range(m), desc="geometry directions")
    for i in pbar:
        dir_start = time.time()
        u = U[i]

        theta_plus = theta0 + eps * u
        vector_to_params(model, theta_plus)
        recalibrate_bn(model, bn_loader, device, num_batches=bn_batches, reset_stats=False)
        Lp = float(eval_classification(model, val_loader, device, max_batches=num_eval_batches)["loss"])

        theta_minus = theta0 - eps * u
        vector_to_params(model, theta_minus)
        recalibrate_bn(model, bn_loader, device, num_batches=bn_batches, reset_stats=False)
        Lm = float(eval_classification(model, val_loader, device, max_batches=num_eval_batches)["loss"])

        vector_to_params(model, theta0)

        sec = (Lp + Lm - 2.0 * L0) / (eps * eps)
        per_dir.append(float(sec))

        elapsed_dir = time.time() - dir_start
        elapsed_total = time.time() - start_total
        avg_dir = elapsed_total / (i + 1)
        eta = avg_dir * (m - i - 1)

        pbar.set_postfix(
            dir_sec=f"{elapsed_dir:.1f}s",
            eta_min=f"{eta / 60:.1f}",
            last_sec=f"{sec:.3e}",
        )

    kappa_tr = float(np.mean(per_dir)) if len(per_dir) > 0 else float("nan")
    kappa_std = float(np.std(per_dir, ddof=1)) if len(per_dir) > 1 else 0.0

    out_dir = ensure_dir(out_path)
    tag = _short_tag(ckpt_path)
    json_path = Path(out_dir) / f"geometry__{tag}.json"

    out: dict[str, Any] = {
        "ckpt": str(ckpt_path),
        "dataset": dataset_name,
        "model": model_name,
        "device": str(device),
        "alpha": alpha,
        "num_directions": m,
        "eval_batch_size": eval_batch_size,
        "num_eval_batches": num_eval_batches,
        "bn_recalib_batches": bn_batches,
        "theta_norm": theta_norm,
        "epsilon": eps,
        "base": base,
        "kappa_tr": kappa_tr,
        "kappa_tr_std": kappa_std,
        "per_direction": per_dir,
    }
    save_json(json_path, out)

    csv_path = Path(out_dir) / "geometries.csv"
    header = (
        "ckpt,model,dataset,alpha,num_directions,eval_batch_size,num_eval_batches,"
        "theta_norm,epsilon,base_loss,base_acc,kappa_tr,kappa_tr_std"
    )
    row = [
        str(ckpt_path),
        model_name,
        dataset_name,
        f"{alpha:.10g}",
        str(m),
        str(eval_batch_size),
        "" if num_eval_batches is None else str(num_eval_batches),
        f"{theta_norm:.10g}",
        f"{eps:.10g}",
        f"{L0:.10g}",
        f"{float(base['acc']):.10g}",
        f"{kappa_tr:.10g}",
        f"{kappa_std:.10g}",
    ]

    if not csv_path.exists():
        csv_path.write_text(header + "\n", encoding="utf-8")
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(",".join(row) + "\n")

    return json_path
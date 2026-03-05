from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def recalibrate_bn(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    num_batches: int = 50,
) -> None:
    """
    Обновление running_mean / running_var у BatchNorm слоёв под текущие веса модели.
    Модель в eval, но BN-слои в train (обновятся running stats без dropout).
    """
    bn_layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bn_layers or num_batches <= 0 or loader is None:
        return

    model.eval()
    for m in bn_layers:
        m.train()

    batches = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        batches += 1
        if batches >= num_batches:
            break

    model.eval()
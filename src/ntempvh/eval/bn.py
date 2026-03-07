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
    reset_stats: bool = False,
) -> None:
    """
    Пересчет running_mean / running_var у BatchNorm-слоев.

    reset_stats=True:
        пересчет stats с нуля (через cumulative average, momentum=None)

    reset_stats=False:
        подстройка от текущих stats
    """
    bn_layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bn_layers or num_batches <= 0 or loader is None:
        return

    was_training = model.training
    saved_momenta = {}

    model.eval()
    for m in bn_layers:
        saved_momenta[m] = m.momentum
        if reset_stats:
            m.reset_running_stats()
            m.momentum = None
        m.train()

    batches = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        batches += 1
        if batches >= num_batches:
            break

    for m in bn_layers:
        m.momentum = saved_momenta[m]

    model.train(was_training)
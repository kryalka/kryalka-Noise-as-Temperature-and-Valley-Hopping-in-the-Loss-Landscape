from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    bn: DataLoader 


def _build_transforms(*, train: bool) -> transforms.Compose:
    """
    Сбор пайплайна преобразований для train (с аугментацией) или val/test (без аугментации)
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _seed_worker(worker_id: int) -> None:
    """
    Фиксация seed numpy/random в воркере DataLoader для воспроизводимости
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_cifar10_loaders(
    root: str,
    batch_size: int,
    *,
    val_size: int = 5000,
    split_seed: int = 0,
    shuffle_seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    val_batch_size: int = 256,
    bn_batch_size: int | None = None
) -> DataLoaders:
    """
    Возвращает train/val DataLoader'ы CIFAR-10 с фиксированным разбиением 
    """

    bn_bs = int(bn_batch_size) if bn_batch_size is not None else int(val_batch_size)


    tf_train = _build_transforms(train=True)
    tf_eval = _build_transforms(train=False)

    train_full_aug = datasets.CIFAR10(root=root, train=True, download=True, transform=tf_train)
    train_full_noaug = datasets.CIFAR10(root=root, train=True, download=True, transform=tf_eval)

    n = len(train_full_aug)

    g_split = torch.Generator()
    g_split.manual_seed(int(split_seed))
    perm = torch.randperm(n, generator=g_split).tolist()

    if val_size == 0:
        train_idx = perm
        val_idx: list[int] = []
    else:
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

    train_ds = Subset(train_full_aug, train_idx)
    val_ds = Subset(train_full_noaug, val_idx)
    bn_ds = Subset(train_full_noaug, train_idx)

    if shuffle_seed is None:
        shuffle_seed = int(split_seed)

    g_loader = torch.Generator()
    g_loader.manual_seed(int(shuffle_seed))

    use_workers = int(num_workers) > 0
    worker_init = _seed_worker if use_workers else None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        worker_init_fn=worker_init,
        generator=g_loader,
        persistent_workers=use_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(val_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        worker_init_fn=worker_init,
        generator=g_loader,
        persistent_workers=use_workers,
    )

    bn_loader = DataLoader(
        bn_ds,
        batch_size=bn_bs,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        worker_init_fn=worker_init,
        generator=g_loader,
        persistent_workers=use_workers,
    )

    return DataLoaders(train=train_loader, val=val_loader, bn=bn_loader)


def get_cifar10_test_loader(
    root: str,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Возвращает test DataLoader CIFAR-10 (без аугментаций) с фиксированным seed
    """
    tf_eval = _build_transforms(train=False)
    test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tf_eval)

    g_loader = torch.Generator()
    g_loader.manual_seed(0)

    use_workers = int(num_workers) > 0
    worker_init = _seed_worker if use_workers else None

    return DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        worker_init_fn=worker_init,
        generator=g_loader,
        persistent_workers=use_workers,
    )
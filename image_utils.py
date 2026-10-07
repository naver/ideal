# Some parts are taken from https://github.com/zkkli/PSAQ-ViT/blob/main/generate_data.py


import numpy as np
import torch
import math

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def standard_normalize(
    data,
    mean_ema=None,
    std_ema=None,
    ema_momentum=0.1,
    eps=1e-4,
):
    """
    Standard normalization with optional EMA stats.
    Gradients flow through normalization; EMA updates are done without grad.
    """
    ndims = data.ndim
    assert ndims in (2, 3)

    dims = [0]
    if ndims == 3:
        dims.append(1)

    mean = data.mean(dim=dims, keepdim=True)
    std = data.std(dim=dims, keepdim=True) + eps

    if mean_ema is None:
        return (data - mean) / std

    mean_ema = mean_ema.to(data.device)
    std_ema = std_ema.to(data.device)

    out = (data - mean_ema) / (std_ema + eps)

    with torch.no_grad():
        mean_ema.mul_(1 - ema_momentum).add_(mean * ema_momentum)
        std_ema.mul_(1 - ema_momentum).add_(std * ema_momentum)

    return out

def unnormalize_for_plot(img_batch):
    mean = torch.tensor(IMAGENET_MEAN, device=img_batch.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img_batch.device).view(1, 3, 1, 1)
    return (img_batch * std + mean).clamp(0, 1)


def clip(image_tensor):
    for c in range(3):
        m, s = IMAGENET_MEAN[c], IMAGENET_STD[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    """Cosine-decay LR schedule with linear warm-up."""
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr
    return lr_policy(_lr_fn)

def build_tnorm_ema_schedule(niter_per_ep: int):
    """Build a cosine EMA schedule for target normalization."""
    return cosine_scheduler(
        1.0,
        0.001,
        epochs=1,
        niter_per_ep=niter_per_ep,
        warmup_epochs=0,
    )

def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    """
    Creates a cosine scheduler with linear warm-up.
    """
    warmup_schedule = np.array([])
    warmup_iters = math.floor(warmup_epochs * niter_per_ep)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(math.floor(epochs * niter_per_ep) - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == math.floor(epochs * niter_per_ep)
    return schedule
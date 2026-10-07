import os
import re
import random

import torch
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from config import Config
from hooks import register_attention_hooks, remove_hooks
from image_utils import unnormalize_for_plot, clip, lr_cosine_policy, standard_normalize, build_tnorm_ema_schedule
from losses import (
    patch_decorrelation_loss,
    image_decorrelation_loss,
    tv_loss,
)
from teachers.prepare import prepare_teacher_models

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

JITTER_LIM = 15

def generation_loss_for_teacher(teacher_model, teacher_stat, tnorm_ema_schedule, images, loss_cfg):
    """
    Compute the teacher-specific decorrelation losses on a batch of images.

    Returns:
        (loss, loss_pd, loss_id)
    """
    teacher_model.zero_grad()
    hooks = register_attention_hooks(teacher_model) if loss_cfg.alpha_pd > 0 else []

    try:
        out_dict = teacher_model.forward_features(images)
        
        loss_pd = patch_decorrelation_loss(hooks)

        global_feat = out_dict["x_norm_clstoken"]
        global_feat = standard_normalize(
                global_feat,
                mean_ema=teacher_stat["cls"]["mean"],
                std_ema=teacher_stat["cls"]["std"],
                ema_momentum=tnorm_ema_schedule,
            )

        loss_id = image_decorrelation_loss(global_feat)

        loss = loss_cfg.alpha_pd * loss_pd + loss_cfg.alpha_id * loss_id
        return loss, loss_pd, loss_id
    finally:
        remove_hooks(hooks)

# Some parts are taken from https://github.com/zkkli/PSAQ-ViT/blob/main/generate_data.py

def optimize_pool(cfg: Config, teacher_models, teachers_ft_stats, img_batch: torch.Tensor):
    """
    Args:
        cfg: cfg.runtime.dir_path must point to the directory of this pool
        teacher_models: dict of hook-injected teacher models
        teachers_ft_stats: feature statistics for each teacher
        img_batch: seed images, shape (pool_size, 3, H, W)

    Returns:
        the optimized images
    """
    device = cfg.runtime.device
    if img_batch is None:
        raise ValueError("img_batch must be provided.")

    img = img_batch.to(device, non_blocking=True)
    img.requires_grad_(True)

    optimizer = optim.Adam([img], lr=cfg.runtime.lr, betas=[0.5, 0.9], eps=1e-8)
    lr_scheduler = lr_cosine_policy(cfg.runtime.lr, 100, cfg.runtime.iterations)

    tv_target = random.uniform(
        2500 * (cfg.runtime.subset_size / 32),
        3000 * (cfg.runtime.subset_size / 32),
    )

    assert cfg.runtime.dir_path is not None, "dir_path must be set before optimize_pool"

    tnorm_ema_schedule = build_tnorm_ema_schedule(cfg.runtime.iterations)


    with tqdm(range(cfg.runtime.iterations), desc="Optimizing pool") as pbar:
        for itr in pbar:
            lr_scheduler(optimizer, itr, itr)

            if itr % cfg.runtime.resample_every == 0:
                idx = torch.randperm(cfg.runtime.pool_size, device=img.device)[: cfg.runtime.subset_size]

            img_subset = img.index_select(0, idx)

            # Random spatial jitter + horizontal flip.
            off = random.randint(-JITTER_LIM, JITTER_LIM)
            img_jit = torch.roll(img_subset, shifts=(off, off), dims=(2, 3))
            if random.random() > 0.5:
                img_jit = torch.flip(img_jit, dims=(3,))

            optimizer.zero_grad(set_to_none=True)

            loss_tv = torch.norm(tv_loss(img_jit) - tv_target) * cfg.loss.alpha_tv

            loss_teachers = 0.0
            teacher_losses = {}
            for name, teacher in teacher_models.items():
                loss_t, loss_pd, loss_id = generation_loss_for_teacher(teacher, teachers_ft_stats[name], tnorm_ema_schedule[itr], img_jit, cfg.loss)
                loss_teachers = loss_teachers + loss_t
                teacher_losses[name] = loss_t

            total_loss = loss_teachers + loss_tv

            total_loss.backward()
            optimizer.step()
            img.data = clip(img.data)

            postfix = dict(
                loss=f"{total_loss.item():.4f}",
                decorr=f"{loss_teachers.item():.4f}",
                tv=f"{loss_tv.item():.4f}",
            )
            for name in teacher_losses:
                short = name.split("_")[0]
                t_loss = teacher_losses[name].item()
                postfix[f"{short}_loss"] = f"{t_loss:.4f}"
            pbar.set_postfix(postfix)

    return img.detach()


def generate_dataset(cfg: Config, teachers, teachers_ft_stats, data_loader: DataLoader):

    total_to_generate = cfg.data.num_images
    device = cfg.runtime.device

    teacher_models = prepare_teacher_models(teachers=teachers, device=device)

    save_root = os.path.join(cfg.runtime.dir_path, "images")
    os.makedirs(save_root, exist_ok=True)

    total_saved = 0
    existing_images = sorted(Path(save_root).glob("image_*.png"))
    if existing_images:
        match = re.search(r"image_(\d+)\.png", existing_images[-1].name)
        if match:
            total_saved = int(match.group(1)) + 1
        print(f"Resuming: {total_saved} images already in {save_root}")

    full_loader = data_loader
    if total_saved > 0:
        remaining = list(range(total_saved, len(data_loader.dataset)))
        data_loader = DataLoader(
            Subset(data_loader.dataset, remaining),
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
        )

    global_image_idx = total_saved
    data_iter = iter(data_loader)
    while total_saved < total_to_generate:
        try:
            img_batch, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(full_loader)
            img_batch, _ = next(data_iter)

        print(f"\n--- Pool starting at image #{total_saved + 1} ---")

        generated = optimize_pool(cfg, teacher_models, teachers_ft_stats, img_batch)

        for i in range(generated.size(0)):
            image = unnormalize_for_plot(generated[i].unsqueeze(0))
            img_path = os.path.join(save_root, f"image_{global_image_idx:08d}.png")
            vutils.save_image(image, img_path)

            global_image_idx += 1
            total_saved += 1
            if total_saved >= total_to_generate:
                break

        print(f"Total saved: {total_saved}/{total_to_generate}")

    print(f"\nDone! All {total_saved} images saved to: {save_root}")

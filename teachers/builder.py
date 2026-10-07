
#Taken from https://github.com/naver/unic/blob/main/teachers/builder.py

import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch

from .config import TEACHER_CFG


def build_teachers(
    teacher_names: List[str],
) -> Tuple[Dict[str, torch.nn.Module], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    teachers = OrderedDict()
    teacher_ft_stats = OrderedDict()

    for tname in teacher_names:
        print("Loading teacher '{}'".format(tname))
        teachers[tname] = _build_teacher(tname)
        teacher_ft_stats[tname] = {}

        # buffers for teacher feature statistics
        ft_dim = TEACHER_CFG[tname]["num_features"]
        token_types = TEACHER_CFG[tname]["token_types"]
        for token_type in token_types:
            dims = (1, ft_dim) if token_type == "cls" else (1, 1, ft_dim)
            teacher_ft_stats[tname][token_type] = {
                "mean": torch.zeros(*dims),
                "std": torch.ones(*dims),
            }

    return teachers, teacher_ft_stats


def _build_teacher(name):
    if name not in TEACHER_CFG:
        raise ValueError(
            f"Unsupported teacher name: {name} (supported: {list(TEACHER_CFG.keys())})"
        )

    cfg = TEACHER_CFG[name]
    ckpt_path = cfg["ckpt_path"]
    if not os.path.exists(ckpt_path):
        raise ValueError(
            f"Teacher checkpoint not found: {ckpt_path}. "
        )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if cfg["ckpt_key"] and cfg["ckpt_key"] in state_dict:
        state_dict = state_dict[cfg["ckpt_key"]]

    model = cfg["loader"](img_size=cfg["image_size"], patch_size=cfg["patch_size"])
    model.load_state_dict(state_dict, strict=True)

    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


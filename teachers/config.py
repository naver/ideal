# Taken from https://github.com/naver/unic/blob/main/teachers/config.py

import os

from .vit_dbotft import dbotft_vitbase
from .vit_deit3 import deit3_vitbase
from .vit_dino import dino_vitbase

CHECKPOINT_DIR = os.environ.get("IDEAL_CHECKPOINT_DIR", "./checkpoints")

TEACHER_CFG = {
    "dino_vitbase_16": {
        "loader": dino_vitbase,
        "ckpt_path": os.path.join(CHECKPOINT_DIR, "dino_vitbase_16.pth"),
        "ckpt_key": "model",
        "num_features": 768,
        "image_size": 224,
        "patch_size": 16,
        "token_types": ["cls", "patch"],
    },
    "deit3_vitbase_16": {
        "loader": deit3_vitbase,
        "ckpt_path": os.path.join(CHECKPOINT_DIR, "deit3_vitbase_16.pth"),
        "ckpt_key": "model",
        "num_features": 768,
        "image_size": 224,
        "patch_size": 16,
        "token_types": ["cls", "patch"],
    },
    "ibot_vitbase_16": {
        # iBOT uses the same ViT implementation as DINO
        "loader": dino_vitbase,
        "ckpt_path": os.path.join(CHECKPOINT_DIR, "ibot_vitbase_16.pth"),
        "ckpt_key": "state_dict",
        "num_features": 768,
        "image_size": 224,
        "patch_size": 16,
        "token_types": ["cls", "patch"],
    },
    "dbotft_vitbase_16": {
        "loader": dbotft_vitbase,
        "ckpt_path": os.path.join(CHECKPOINT_DIR, "dbotft_vitbase_16.pth"),
        "ckpt_key": "model",
        "num_features": 768,
        "image_size": 224,
        "patch_size": 16,
        "token_types": ["cls", "patch"],
    },
}

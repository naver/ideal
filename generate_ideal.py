import argparse
import os
import random

import numpy as np
import torch

from config import build_configs, Config
from data_utils import build_dataset
from teachers import build_teachers
from optimization import generate_dataset


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def get_args_parser():
    parser = argparse.ArgumentParser(description="IDeaL sample generation", add_help=False)

    # data
    parser.add_argument("--dataset_path", default=None,
                        help="Seed images: a directory of images, or a .pkl file with image paths. "
                             "Required unless --dataset_type noise (seeds generated on the fly).")
    parser.add_argument("--dataset_type", default="dead_leaves",
                        choices=["dead_leaves", "noise"],
                        help="Type of seed images used to initialize the optimization")
    parser.add_argument("--num_images", default=1000, type=int,
                        help="Total number of IDeaL samples to generate")
    parser.add_argument("--output_dir", required=True,
                        help="Root directory for generated images and diagnostics")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Size of generated images")

    # teachers
    parser.add_argument("--teachers", type=str,
                        default="dino_vitbase_16,deit3_vitbase_16,ibot_vitbase_16,dbotft_vitbase_16",
                        help="Comma-separated list of teacher names (see teachers/config.py)")

    parser.add_argument("--pool_size", default=250, type=int,
                        help="Number of seed images optimized jointly per pool")
    parser.add_argument("--subset_size", default=40, type=int,
                        help="Random subset of the pool optimized at each iteration")
    parser.add_argument("--resample_every", default=10, type=int,
                        help="Iterations between subset re-draws")
    parser.add_argument("--iterations", default=4000, type=int,
                        help="Pixel-optimization iterations per pool")
    parser.add_argument("--lr", default=0.1, type=float)

    parser.add_argument("--alpha_tv", default=0.05, type=float)
    parser.add_argument("--alpha_pd", default=1.0, type=float)
    parser.add_argument("--alpha_id", default=1.0, type=float)

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    return parser


def main(cfg: Config):
    set_seeds(cfg.runtime.seed)

    print("Loading teachers ...")
    teachers, teachers_ft_stats = build_teachers(cfg.model.teachers)

    os.makedirs(cfg.runtime.dir_path, exist_ok=True)

    print(f"Loading seed dataset ({cfg.data.dataset_type}) ...")
    train_loader, train_dataset = build_dataset(cfg)
    print(f"Loaded dataset:\n - {train_dataset}")

    print("Starting IDeaL sample generation ...")
    generate_dataset(cfg, teachers, teachers_ft_stats, train_loader)
    print("Sample generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("IDeaL", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.dataset_type != "noise" and args.dataset_path is None:
        parser.error("--dataset_path is required unless --dataset_type noise")
    cfg = build_configs(args)
    main(cfg)

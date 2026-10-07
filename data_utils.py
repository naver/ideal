import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import VisionDataset

from config import Config
from image_utils import IMAGENET_MEAN, IMAGENET_STD


def pil_loader(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class PickleSubset(VisionDataset):

    def __init__(self, subset_path, transform=None, imroot=""):
        with open(subset_path, "rb") as f:
            self.samples = pickle.load(f)
        self.loader = pil_loader
        self.transform = transform
        self.transforms = transform
        self.root = imroot

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if isinstance(sample, tuple):
            image_path, target = sample
        else:
            image_path, target = sample, 0
        image_path = self.root + image_path

        try:
            image = self.loader(image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}") from e

        image = self.transform(image) if self.transform else image
        return image, target


class NoiseDataset(VisionDataset):
    """Gaussian-noise seed images, generated on first use and cached on disk."""

    def __init__(self, noise_dir, num_images, img_size, transform=None):
        self.noise_dir = Path(noise_dir)
        self.root = noise_dir
        self.transform = transform
        self.transforms = transform
        self.loader = pil_loader

        self._generate_noise_images(num_images, img_size)
        noise_images = sorted(self.noise_dir.glob("noise_*.png"))
        self.samples = [(str(img), 0) for img in noise_images]

    def _generate_noise_images(self, num_images, img_size):
        os.makedirs(self.noise_dir, exist_ok=True)
        existing = list(self.noise_dir.glob("noise_*.png"))
        if len(existing) >= num_images:
            return
        start_idx = len(existing)
        for i in range(start_idx, num_images):
            noise = np.random.randn(img_size, img_size, 3)
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
            noise = (noise * 255).astype(np.uint8)
            Image.fromarray(noise, mode="RGB").save(self.noise_dir / f"noise_{i:06d}.png")
        print(f"Generated {num_images - start_idx} noise images in {self.noise_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        try:
            image = self.loader(image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}") from e
        if self.transform:
            image = self.transform(image)
        return image, target


class FlatImageFolder(VisionDataset):

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.transforms = transform
        self.loader = pil_loader

        root_path = Path(root)
        self.samples = sorted(
            (str(p), 0) for p in root_path.iterdir()
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        try:
            image = self.loader(image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}") from e
        if self.transform:
            image = self.transform(image)
        return image, target


def build_dataset(cfg: Config):
    train_transform = build_transform(input_size=cfg.data.img_size)

    if cfg.data.dataset_type == "noise":
        noise_dir = Path(cfg.runtime.dir_path) / "noise_images"
        train_dataset = NoiseDataset(
            noise_dir=str(noise_dir),
            num_images=cfg.data.num_images,
            img_size=cfg.data.img_size,
            transform=train_transform,
        )
    elif cfg.data.dataset_path.endswith(".pkl"):
        train_dataset = PickleSubset(cfg.data.dataset_path, train_transform)
    else:
        data_path = Path(cfg.data.dataset_path)
        if not data_path.is_dir():
            raise ValueError(f"Dataset path is not a directory: {data_path}")
        has_subdirs = any(p.is_dir() for p in data_path.iterdir())
        if has_subdirs:
            train_dataset = datasets.ImageFolder(str(data_path), train_transform)
        else:
            train_dataset = FlatImageFolder(str(data_path), train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.runtime.pool_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, train_dataset


def build_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize(input_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

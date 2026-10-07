from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    seed: int
    iterations: int       
    pool_size: int         
    subset_size: int      
    resample_every: int  
    lr: float
    dir_path: Optional[str] = None


@dataclass(frozen=True)
class DataConfig:
    dataset_path: str
    dataset_type: str      # 'dead_leaves' | 'noise' | 'images'
    num_images: int
    img_size: int = 224


@dataclass(frozen=True)
class ModelConfig:
    teachers: List[str]


@dataclass(frozen=True)
class LossConfig:
    alpha_tv: float
    alpha_pd: float
    alpha_id: float


@dataclass(frozen=True)
class Config:
    runtime: RuntimeConfig
    data: DataConfig
    model: ModelConfig
    loss: LossConfig


def build_configs(args) -> Config:
    runtime = RuntimeConfig(
        device=args.device,
        seed=args.seed,
        iterations=args.iterations,
        pool_size=args.pool_size,
        subset_size=args.subset_size,
        resample_every=args.resample_every,
        lr=args.lr,
        dir_path=args.output_dir,
    )

    data = DataConfig(
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        num_images=args.num_images,
    )

    model = ModelConfig(teachers=sorted(args.teachers.split(",")))

    loss = LossConfig(
        alpha_tv=args.alpha_tv,
        alpha_pd=args.alpha_pd,
        alpha_id=args.alpha_id,
    )

    return Config(runtime, data, model, loss)

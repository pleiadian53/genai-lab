"""Configuration management using Pydantic and Hydra."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data configuration."""

    dataset: str = "toy"
    n_samples: int = 10000
    n_genes: int = 2000
    n_tissues: int = 5
    n_diseases: int = 3
    n_batches: int = 8
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42


class ModelConfig(BaseModel):
    """Model configuration."""

    model_type: str = "cvae"
    z_dim: int = 64
    hidden_dim: int = 512
    n_layers: int = 2
    dropout: float = 0.1
    emb_dim: int = 32
    cond_dim: int = 128


class TrainConfig(BaseModel):
    """Training configuration."""

    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    beta: float = 1.0
    beta_warmup: int = 0
    grad_clip: float = 5.0
    early_stopping: int = 10


class Config(BaseModel):
    """Main configuration."""

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    outdir: str = "runs/experiment"
    seed: int = 42
    device: str = "auto"


def load_config(path: str | Path) -> Config:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Config object
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    return Config(**data)


def save_config(config: Config, path: str | Path):
    """Save configuration to YAML file.

    Args:
        config: Config object
        path: Output path
    """
    import yaml

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)


def config_from_args(args: Any) -> Config:
    """Create config from command line arguments.

    Args:
        args: Parsed arguments (e.g., from argparse)

    Returns:
        Config object
    """
    config = Config()

    # Override with args
    for key, value in vars(args).items():
        if value is not None:
            parts = key.split(".")
            if len(parts) == 1:
                if hasattr(config, key):
                    setattr(config, key, value)
            elif len(parts) == 2:
                section, param = parts
                if hasattr(config, section):
                    section_config = getattr(config, section)
                    if hasattr(section_config, param):
                        setattr(section_config, param, value)

    return config

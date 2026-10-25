"""Typed, YAML-backed experiment configuration.

Every hyper-parameter lives here (or in a YAML file under ``configs/``); nothing is hard-coded
in the pipeline. Any value can be overridden from the CLI: ``--set autoencoder.latent_dim=32``.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml


def _default_dqn_pixel() -> dict:
    return dict(learning_rate=1e-4, batch_size=64, buffer_size=20_000, learning_starts=2000, gamma=0.99,
                target_update_interval=500, train_freq=4, gradient_steps=1,
                exploration_fraction=0.2, exploration_final_eps=0.05)


def _default_dqn_latent() -> dict:  # RL-Zoo style CartPole settings for vector inputs
    return dict(learning_rate=2.3e-3, batch_size=64, buffer_size=100_000, learning_starts=1000, gamma=0.99,
                target_update_interval=10, train_freq=256, gradient_steps=128,
                exploration_fraction=0.16, exploration_final_eps=0.04, net_arch=[256, 256])


@dataclass
class EnvConfig:
    env_id: str = "CartPole-v1"
    image_size: int = 64
    crop_rows: Optional[list] = field(default_factory=lambda: [150, 350])  # None = no crop
    frame_stack: int = 4          # both pipelines stack consecutive observations (velocity cue)


@dataclass
class AutoencoderConfig:
    latent_dim: int = 16
    num_frames: int = 30_000      # unlabelled frames used for training
    diverse_initial_states: bool = True
    data_seed: int = 123
    epochs: int = 30
    learning_rate: float = 1e-3
    batch_size: int = 256
    foreground_loss_weight: float = 10.0  # up-weights non-background pixels (thin pole)
    validation_fraction: float = 0.1


@dataclass
class AgentConfig:
    total_timesteps: int = 100_000
    eval_every_steps: int = 5_000
    eval_episodes: int = 10
    success_threshold: float = 400.0
    dqn_pixel: dict = field(default_factory=_default_dqn_pixel)
    dqn_latent: dict = field(default_factory=_default_dqn_latent)


@dataclass
class ExperimentConfig:
    name: str = "main"
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    output_dir: str = "outputs"
    device: str = "auto"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # ---------- construction ----------
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        merged = _deep_update(asdict(cls()), d or {})
        return cls(env=EnvConfig(**merged["env"]), autoencoder=AutoencoderConfig(**merged["autoencoder"]),
                   agent=AgentConfig(**merged["agent"]), experiment=ExperimentConfig(**merged["experiment"]))

    @classmethod
    def from_yaml(cls, path=None, overrides=()) -> "Config":
        d = {}
        if path:
            with open(path) as f:
                d = yaml.safe_load(f) or {}
        for item in overrides:
            _apply_override(d, item)
        return cls.from_dict(d)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def with_latent_dim(self, dim: int) -> "Config":
        c = copy.deepcopy(self)
        c.autoencoder.latent_dim = dim
        return c

    # ---------- output layout ----------
    @property
    def run_dir(self) -> Path:
        return Path(self.experiment.output_dir) / self.experiment.name

    @property
    def results_dir(self) -> Path:
        return self.run_dir / "results"

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.run_dir / "tables"

    @property
    def frame_dataset_path(self) -> Path:
        a = self.autoencoder
        return (self.run_dir / "data" /
                f"frames_{self.env.env_id}_n{a.num_frames}_seed{a.data_seed}_diverse{int(a.diverse_initial_states)}.npz")

    def autoencoder_dir(self, latent_dim: Optional[int] = None) -> Path:
        return self.run_dir / "autoencoders" / f"latent_dim_{latent_dim or self.autoencoder.latent_dim}"

    @property
    def pixel_method(self) -> str:
        return "pixel_dqn"

    def latent_method(self, latent_dim: Optional[int] = None) -> str:
        return f"latent_dqn_z{latent_dim or self.autoencoder.latent_dim}"


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _apply_override(d: dict, item: str) -> None:
    key, _, raw = item.partition("=")
    parts = key.strip().split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)

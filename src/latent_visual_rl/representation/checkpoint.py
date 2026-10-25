"""Trained-autoencoder container, (de)serialisation and batched encoding."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from latent_visual_rl.representation.autoencoder import ConvAutoencoder


@dataclass
class TrainedAutoencoder:
    model: ConvAutoencoder
    latent_mean: np.ndarray   # per-dimension statistics of z on the training set
    latent_std: np.ndarray    # used to standardise z before it reaches the RL agent
    latent_dim: int

    def freeze(self) -> "TrainedAutoencoder":
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self


def save_autoencoder(trained: TrainedAutoencoder, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=trained.model.state_dict(), latent_mean=trained.latent_mean,
                    latent_std=trained.latent_std, latent_dim=trained.latent_dim), path)


def load_autoencoder(path, device) -> TrainedAutoencoder:
    """Load a checkpoint written by this package OR by the walkthrough notebook
    (the notebook stores the latent statistics under 'z_mean' / 'z_std')."""
    ck = torch.load(path, map_location=device, weights_only=False)
    mean = ck["latent_mean"] if "latent_mean" in ck else ck["z_mean"]
    std = ck["latent_std"] if "latent_std" in ck else ck["z_std"]
    model = ConvAutoencoder(ck["latent_dim"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return TrainedAutoencoder(model, mean, std, ck["latent_dim"]).freeze()


def frames_to_tensor(frames: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().to(device)


@torch.no_grad()
def encode_frames(trained: TrainedAutoencoder, frames, device, batch_size=1024, standardize=True):
    trained.model.eval()
    out = [trained.model.encoder(frames_to_tensor(frames[i:i + batch_size], device).float() / 255.0).cpu().numpy()
           for i in range(0, len(frames), batch_size)]
    z = np.concatenate(out)
    return (z - trained.latent_mean) / trained.latent_std if standardize else z

"""Observation wrappers: rendered pixels, and pixels encoded by a frozen autoencoder."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame (Colab / servers)

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


def preprocess_frame(frame, image_size, crop_rows=None):
    """RGB frame -> (image_size, image_size, 1) uint8 grayscale."""
    if crop_rows is not None:
        frame = frame[crop_rows[0]:crop_rows[1]]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return gray[..., None]


class PixelObservation(gym.ObservationWrapper):
    """Replace the low-dimensional state with a rendered grayscale image."""

    def __init__(self, env, image_size, crop_rows):
        super().__init__(env)
        self.image_size, self.crop_rows = image_size, crop_rows
        self.observation_space = spaces.Box(0, 255, (image_size, image_size, 1), np.uint8)

    def observation(self, obs):
        return preprocess_frame(self.env.render(), self.image_size, self.crop_rows)


class LatentObservation(gym.ObservationWrapper):
    """Pixels -> frozen encoder -> standardised latent vector z."""

    def __init__(self, env, trained_autoencoder, device):
        super().__init__(env)
        self.encoder = trained_autoencoder.model.encoder
        self.device = device
        self.mean = trained_autoencoder.latent_mean.astype(np.float32)
        self.std = trained_autoencoder.latent_std.astype(np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (trained_autoencoder.latent_dim,), np.float32)

    @torch.no_grad()
    def observation(self, obs):
        x = torch.from_numpy(obs).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        z = self.encoder(x)[0].cpu().numpy()
        return ((z - self.mean) / self.std).astype(np.float32)

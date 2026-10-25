"""Draw CartPole at an arbitrary state and push it through an autoencoder."""
import threading

import gymnasium as gym
import numpy as np
import torch

from latent_visual_rl.environments.observation_wrappers import preprocess_frame

LIMIT_CART_POS = 2.4        # CartPole terminates beyond |x| = 2.4
LIMIT_POLE_DEG = 12.0       # ... and beyond ~12 degrees of pole tilt


class StateRenderer:
    """Thread-safe renderer: set a physical state, get the agent's (64,64,1) uint8 observation."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._env = gym.make(cfg.env.env_id, render_mode="rgb_array")
        self._env.reset(seed=0)
        self._lock = threading.Lock()      # Gradio calls handlers from worker threads

    def observe(self, cart_pos, pole_angle_deg, cart_vel=0.0, pole_vel=0.0):
        state = np.array([cart_pos, cart_vel, np.deg2rad(pole_angle_deg), pole_vel])
        with self._lock:
            self._env.unwrapped.state = state
            frame = self._env.render()
        return preprocess_frame(frame, self.cfg.env.image_size, self.cfg.env.crop_rows)


@torch.no_grad()
def encode_decode(ae: "TrainedAutoencoder", frame: np.ndarray):
    """frame (H,W,1) uint8 -> (reconstruction in [0,1] as (H,W), standardised latent vector z)."""
    x = torch.from_numpy(frame).permute(2, 0, 1).float().div(255).unsqueeze(0)
    recon, z = ae.model.eval()(x)
    return recon[0, 0].numpy(), (z[0].numpy() - ae.latent_mean) / ae.latent_std

"""Vectorised environment factory shared by training and evaluation."""
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from latent_visual_rl.environments.observation_wrappers import LatentObservation, PixelObservation
from latent_visual_rl.reproducibility import resolve_device


def make_vec_env(cfg, observation_mode: str, seed: int, trained_autoencoder=None):
    """observation_mode in {'pixel', 'latent'}; returns a frame-stacked VecEnv with one env."""
    if observation_mode not in ("pixel", "latent"):
        raise ValueError(f"unknown observation_mode: {observation_mode}")
    device = resolve_device(cfg.experiment.device)

    def _make():
        env = gym.make(cfg.env.env_id, render_mode="rgb_array")
        env = PixelObservation(env, cfg.env.image_size, cfg.env.crop_rows)
        if observation_mode == "latent":
            env = LatentObservation(env, trained_autoencoder, device)
        return Monitor(env)

    venv = DummyVecEnv([_make])
    venv.seed(seed)
    venv = VecFrameStack(venv, cfg.env.frame_stack)
    if observation_mode == "pixel":
        venv = VecTransposeImage(venv)  # HWC -> CHW for CnnPolicy
    return venv

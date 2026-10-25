import numpy as np

from latent_visual_rl.config import Config
from latent_visual_rl.environments.factory import make_vec_env
from latent_visual_rl.representation.autoencoder import ConvAutoencoder
from latent_visual_rl.representation.checkpoint import TrainedAutoencoder


def test_pixel_env_observation_shape():
    env = make_vec_env(Config(), "pixel", seed=0)
    obs = env.reset()
    assert obs.shape == (1, 4, 64, 64) and obs.dtype == np.uint8   # CHW, 4 stacked frames


def test_latent_env_observation_shape():
    cfg = Config()
    ae = TrainedAutoencoder(ConvAutoencoder(16),
                            np.zeros(16), np.ones(16), 16).freeze()
    env = make_vec_env(cfg, "latent", seed=0, trained_autoencoder=ae)
    assert env.reset().shape == (1, 16 * cfg.env.frame_stack)

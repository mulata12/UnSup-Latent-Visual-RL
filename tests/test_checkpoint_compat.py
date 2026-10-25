import numpy as np
import torch

from latent_visual_rl.representation.autoencoder import ConvAutoencoder
from latent_visual_rl.representation.checkpoint import TrainedAutoencoder, load_autoencoder, save_autoencoder


def test_roundtrip_package_format(tmp_path):
    ae = TrainedAutoencoder(ConvAutoencoder(8), np.zeros(8), np.ones(8), 8)
    save_autoencoder(ae, tmp_path / "a.pt")
    assert load_autoencoder(tmp_path / "a.pt", "cpu").latent_dim == 8


def test_loads_notebook_format(tmp_path):
    # The walkthrough notebook saves 'z_mean' / 'z_std' (+ history) instead of 'latent_mean' / 'latent_std'.
    torch.save(dict(state_dict=ConvAutoencoder(4).state_dict(), z_mean=np.zeros(4), z_std=np.ones(4),
                    latent_dim=4, history={}, pretrain_seconds=1.0), tmp_path / "nb.pt")
    ae = load_autoencoder(tmp_path / "nb.pt", "cpu")
    assert ae.latent_dim == 4 and ae.latent_std.shape == (4,)

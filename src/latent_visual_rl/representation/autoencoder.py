"""Convolutional autoencoder architecture (64x64 -> latent_dim -> 64x64)."""
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int, in_channels: int = 1):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))

        self.conv = nn.Sequential(block(in_channels, 32), block(32, 64), block(64, 128), block(128, 256))
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int = 1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        def block(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))

        self.deconv = nn.Sequential(block(256, 128), block(128, 64), block(64, 32),
                                    nn.ConvTranspose2d(32, out_channels, 4, 2, 1), nn.Sigmoid())

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 256, 4, 4))


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int, in_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(latent_dim, in_channels)
        self.decoder = ConvDecoder(latent_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

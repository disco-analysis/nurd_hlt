"""Autoencoder for object-level HLT features (axis 1 of ABCD)."""
import torch
import torch.nn as nn


class _Encoder(nn.Module):
    def __init__(self, features, hidden_nodes, latent_dim):
        super().__init__()
        layers, in_dim = [], features
        for node in hidden_nodes:
            layers += [nn.Linear(in_dim, node), nn.ReLU(inplace=True)]
            in_dim = node
        self.backbone = nn.Sequential(*layers)
        self.fc_latent = nn.Linear(in_dim, latent_dim)

    def forward(self, x):
        return self.fc_latent(self.backbone(x))


class _Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_nodes):
        super().__init__()
        layers, in_dim = [], latent_dim
        for i, node in enumerate(hidden_nodes):
            lin = nn.Linear(in_dim, node)
            if i == len(hidden_nodes) - 1:
                nn.init.uniform_(lin.weight, -0.05, 0.05)
                nn.init.uniform_(lin.bias,   -0.05, 0.05)
            layers.append(lin)
            if i != len(hidden_nodes) - 1:
                layers += [nn.BatchNorm1d(node), nn.ReLU(inplace=True)]
            in_dim = node
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class HLTAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha   = float(config.get("alpha", 1.0))
        features     = config["features"]
        latent_dim   = config["latent_dim"]
        enc_nodes    = config["encoder_config"]["nodes"]
        dec_nodes    = config["decoder_config"]["nodes"]
        self.encoder = _Encoder(features, enc_nodes, latent_dim)
        self.decoder = _Decoder(latent_dim, dec_nodes)

    def forward(self, x):
        z     = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

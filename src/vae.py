# -*- coding: utf-8 -*-
"""
Autor: Matheus Becali Rocha
Email: matheusbecali@gmail.com
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder profundo
# class Encoder(nn.Module):
#     """
#     Encoder profundo e regularizado para um VAE.
#     """
#     def __init__(self, input_dim=8, latent_dim=16, dropout_rate=0.1):
#         super().__init__()

#         self.encoder_base = nn.Sequential(
#             nn.Linear(input_dim, 20),
#             nn.BatchNorm1d(20),
#             nn.Dropout(dropout_rate),
#             # nn.ReLU(),
#             nn.ELU(),
#             # nn.Tanh(),
#             nn.Linear(20, 16),
#             nn.BatchNorm1d(16),
#             nn.Dropout(dropout_rate),
#             # nn.ReLU(),
#             nn.ELU(),
#             # nn.Tanh(),
#             nn.Linear(16, 14),
#             nn.BatchNorm1d(14),
#             nn.Dropout(dropout_rate),
#             # nn.ReLU(),
#             nn.ELU(),
#             # nn.Tanh(),
#             nn.Linear(14, 12),
#             nn.BatchNorm1d(12),
#             nn.Dropout(dropout_rate),
#             # nn.ReLU(),
#             nn.ELU(),
#             # nn.Tanh(),
#             nn.Linear(12, 10),
#             # nn.Sigmoid()
#             # nn.ReLU()
#             nn.ELU(),            
#             # nn.Tanh()
#         )

#         self.fc_mean = nn.Linear(10, latent_dim)
#         self.fc_logvar = nn.Linear(10, latent_dim)

#     def forward(self, x):
#         # 1. Passa a entrada pela rede base
#         h = self.encoder_base(x)

#         # 2. Calcula mu e log_var a partir da representação oculta final 'h'
#         mean = self.fc_mean(h)
#         log_var = self.fc_logvar(h)

#         return mean, log_var

# # Decoder profundo
# class Decoder(nn.Module):
#     """
#     Decoder profundo e regularizado para um VAE.
#     """
#     def __init__(self, latent_dim=16, output_dim=8, dropout_rate=0.1):
#         super().__init__()

#         self.decoder_full = nn.Sequential(
#             nn.Linear(latent_dim, 12),
#             nn.BatchNorm1d(12),
#             nn.Dropout(dropout_rate),
#             # nn.ReLU(),
#             nn.ELU(),
#             # nn.Tanh(), 
#             nn.Linear(12, 14),
#             nn.BatchNorm1d(14),
#             nn.Dropout(dropout_rate),
#             # nn.ReLU(),
#             nn.ELU(),
#             # nn.Tanh(),
#             nn.Linear(14, 16),
#             nn.BatchNorm1d(16),
#             nn.Dropout(dropout_rate),
#             # nn.ReLU(),
#             nn.ELU(),
#             # nn.Tanh(),
#             nn.Linear(16, 20),
#             nn.BatchNorm1d(20),
#             nn.Dropout(dropout_rate),
#             nn.ELU(),
#             # nn.ReLU(),
#             # nn.Tanh(),
#             nn.Linear(20, output_dim),
#             # nn.Sigmoid()
#         )

#     def forward(self, z):
#         return self.decoder_full(z)

class MixedAdversary(nn.Module):
    def __init__(self, latent_dim, attribute_types, hidden_dim=6):
        super(MixedAdversary, self).__init__()

        self.branches = nn.ModuleList()
        for _ in attribute_types:
            # Camada de entrada, camadas ocultas, e camada de saída
            layers = [
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                # nn.ReLU(),
                nn.ELU(),
                # nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                # nn.ReLU(),
                nn.ELU(),
                # nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            ]
            self.branches.append(nn.Sequential(*layers))

    def forward(self, z):
        return [branch(z) for branch in self.branches]

####################################################################################################

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )
    def forward(self, z):
        return self.net(z)

# class MixedAdversary(nn.Module):
#     def __init__(self, latent_dim, attribute_types, hidden_dim=8):
#         super().__init__()
#         self.attribute_types = attribute_types
#         self.backbone = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
#         )
#         heads = []
#         for t in attribute_types:
#             if t == 'binary':
#                 heads.append(nn.Linear(hidden_dim, 1))
#             elif t in ('regression', 'continuous'):
#                 heads.append(nn.Linear(hidden_dim, 1))
#             else:
#                 raise ValueError(f"attribute type não suportado: {t}")
#         self.heads = nn.ModuleList(heads)
#     def forward(self, z):
#         h = self.backbone(z)
#         return [head(h) for head in self.heads]

# class Encoder(nn.Module):
#     def __init__(self, input_dim, latent_dim, hidden=256):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden), nn.ReLU(),
#             nn.Linear(hidden, hidden), nn.ReLU(),
#             nn.Linear(hidden, latent_dim)
#         )
#     def forward(self, x):
#         return self.net(x)

# class Decoder(nn.Module):
#     def __init__(self, latent_dim, input_dim, hidden=256):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(latent_dim, hidden), nn.ReLU(),
#             nn.Linear(hidden, hidden), nn.ReLU(),
#             nn.Linear(hidden, input_dim)
#         )
#     def forward(self, z):
#         return self.net(z)

# class MixedAdversary(nn.Module):
#     def __init__(self, latent_dim, attribute_types, hidden_dim=128):
#         super().__init__()
#         self.attribute_types = attribute_types
#         self.backbone = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
#         )
#         self.heads = nn.ModuleList([
#             nn.Linear(hidden_dim, 1) for _ in attribute_types
#         ])
#     def forward(self, z):
#         h = self.backbone(z)
#         return [head(h) for head in self.heads]

####################################################################################################

def reparameterize(mean, log_var):
    std = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(std)
    return mean + epsilon * std

def vae_losses(recon, x, mu, logvar, recon_type='mse'):
    # perdas como MÉDIA por amostra
    if recon_type == 'mse':
        recon_loss = F.mse_loss(recon, x, reduction='mean')
    elif recon_type == 'bce':
        # assume recon são logits e x in [0,1]
        recon_loss = F.binary_cross_entropy_with_logits(recon, x, reduction='mean')
    else:
        raise ValueError("recon_type deve ser 'mse' ou 'bce'")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl

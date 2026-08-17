# -*- coding: utf-8 -*-
"""
Building blocks of the adversarial variational autoencoder.

Holds the Encoder / Decoder pair, the MixedAdversary that tries to recover the
sensitive attributes from the latent space, and the two functions the training
loop needs: reparameterize (the VAE reparameterization trick) and vae_losses
(reconstruction + KL divergence).

Author: Matheus Becali Rocha
Email: matheusbecali@gmail.com
"""


import torch
import torch.nn.functional as F
from torch import nn


class MixedAdversary(nn.Module):
    """
    Adversary that predicts the sensitive attributes from the latent space.

    One independent branch per attribute, so a binary and a continuous attribute
    can be handled side by side. Every branch emits a single raw value: a logit
    for binary attributes, a prediction for regression ones. The training loop
    picks the matching loss based on `attribute_types`.

    Args:
        latent_dim: Dimensionality of the latent vector z.
        attribute_types: One entry per sensitive attribute; only its length is
            used here, the type itself is applied when choosing the loss.
        hidden_dim: Width of the hidden layers of each branch.
    """
    def __init__(self, latent_dim, attribute_types, hidden_dim=6):
        super(MixedAdversary, self).__init__()

        self.branches = nn.ModuleList()
        for _ in attribute_types:
            # Input layer, hidden layers and output layer
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
        """Return one prediction per sensitive attribute, as a list of tensors."""
        return [branch(z) for branch in self.branches]

####################################################################################################

class Encoder(nn.Module):
    """
    VAE encoder: maps the input to the parameters of the latent distribution.

    Two shared hidden layers feed two separate heads, one for the mean and one
    for the log-variance, so the latent vector z can be sampled by
    reparameterize().

    Args:
        input_dim: Number of input features.
        latent_dim: Dimensionality of the latent space.
        hidden: Width of the hidden layers.
    """
    def __init__(self, input_dim, latent_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
    def forward(self, x):
        """Return the (mean, log_var) pair describing the latent distribution."""
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    """
    VAE decoder: reconstructs the input features from a latent vector.

    Args:
        latent_dim: Dimensionality of the latent space.
        input_dim: Number of features to reconstruct.
        hidden: Width of the hidden layers.
    """
    def __init__(self, latent_dim, input_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )
    def forward(self, z):
        """Return the reconstruction; the output layer is linear, with no activation."""
        return self.net(z)

####################################################################################################

def reparameterize(mean, log_var):
    """
    Sample the latent vector z using the reparameterization trick.

    Sampling z directly would break backpropagation, so the randomness is moved
    into an independent epsilon: z = mean + epsilon * std. The gradient then
    flows through mean and log_var.

    Args:
        mean: Mean of the latent distribution, as returned by the encoder.
        log_var: Log-variance of the latent distribution.

    Returns:
        The sampled latent vector z.
    """
    std = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(std)
    return mean + epsilon * std

def vae_losses(recon, x, mu, logvar, recon_type='mse'):
    """
    Compute the two terms of the VAE loss.

    They are returned separately so the training loop can weight the KL term by
    its own beta (as in a beta-VAE) instead of summing them here.

    Args:
        recon: Reconstruction produced by the decoder.
        x: Original input.
        mu: Mean of the latent distribution.
        logvar: Log-variance of the latent distribution.
        recon_type: 'mse' for continuous data, 'bce' for data in [0,1].

    Returns:
        A tuple (recon_loss, kl), both averaged per sample.

    Raises:
        ValueError: when recon_type is neither 'mse' nor 'bce'.
    """
    # losses as a MEAN per sample
    if recon_type == 'mse':
        recon_loss = F.mse_loss(recon, x, reduction='mean')
    elif recon_type == 'bce':
        # assumes recon holds logits and x lies in [0,1]
        recon_loss = F.binary_cross_entropy_with_logits(recon, x, reduction='mean')
    else:
        raise ValueError("recon_type must be 'mse' or 'bce'")
    # KL divergence between the latent distribution and a standard normal
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl

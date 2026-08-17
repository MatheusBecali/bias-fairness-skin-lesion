# -*- coding: utf-8 -*-
"""
Network architectures and adversarial training loops.

Holds the MLP classifier used by every experiment and the two adversarial
debiasing trainers: train_debiased_vae (variational) and
train_debiased_autoencoder (deterministic). Both alternate two phases per
batch, an encoder/decoder step that tries to fool the adversary and an
adversary step that tries to recover the sensitive attributes from the latent
space, so the representation keeps the useful signal while dropping the
protected information.

Author: Matheus Becali Rocha
Email: matheusbecali@gmail.com
"""

import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim

from src import vae
from utils.helpers import eval_adversary

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
try:
    print(f"Device in use: {torch.cuda.get_device_name(device)}")
except Exception:
    print('No CUDA device found, falling back to CPU.')

# Global seed, shared by every stochastic step so runs are reproducible
_seed = 78645

class EarlyStopper:
    """
    Early stopping: interrupts training once a monitored metric stops improving.
    """
    def __init__(self, patience=10, min_delta=1e-4):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement
                            of the validation metric.
            min_delta (float): Minimum change of the monitored metric for it to
                               count as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop_flag = False

    def early_stop(self, val_loss):
        """
        Record one validation loss and report whether training should stop.

        Args:
            val_loss: Validation loss of the epoch that just finished.

        Returns:
            True once `patience` epochs went by without an improvement.
        """
        # Check whether the validation loss improved
        if val_loss < self.best_loss - self.min_delta:
            # It improved: store the new best loss and reset the counter
            self.best_loss = val_loss
            self.counter = 0
        else:
            # It did not improve: increment the counter
            self.counter += 1
            
        # Once the counter reaches the patience limit, raise the stop flag
        if self.counter >= self.patience:
            self.early_stop_flag = True
            
        return self.early_stop_flag

# Convert to Tensors and prepare data loaders
class ClassifyingNetwork(nn.Module):
    """
    The MLP classifier used by every experiment.

    A single hidden layer with BatchNorm, Dropout and a Tanh activation. The
    commented-out block below keeps the deeper two-hidden-layer variant that was
    tried during development.

    Args:
        input_size: Number of input features.
        hidden_size: Width of the hidden layer.
        num_classes: Number of output classes.
        dropout_rate: Dropout probability of the hidden layer.
    """
    def __init__(self, input_size, hidden_size, num_classes=2, dropout_rate=0.1):
        super(ClassifyingNetwork, self).__init__()

        self.MLPclassify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Tanh(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        """Run the forward pass, returning the raw logits (no softmax)."""
        x = x.to(device)
        output = self.MLPclassify(x)
        return output

def train_debiased_vae(train_loader, val_loader, input_dim, attribute_types, latent_dim=100, 
                       num_epochs=100, lambda_adv=1e-3, beta_vae=1, lr=1e-3, lr_adv=1e-4, 
                       patience=10, class_weights=None, _dataset_name="dataset", mitigation_type="None", model_type="vae", 
                       optimizer="Adam", verbose=True):
    """
    Train a variational autoencoder adversarially, to debias its latent space.

    Each batch runs two phases. Phase 1 updates the encoder/decoder to minimize
    the VAE loss (reconstruction + KL) MINUS the adversary loss, so fooling the
    adversary is rewarded. Phase 2 updates the adversary alone, on a detached
    latent vector, to be as good as it can at recovering the sensitive
    attributes. The equilibrium is a latent space that reconstructs the data
    well while carrying little sensitive information.

    Both beta and lambda_adv are warmed up linearly over the first 100 epochs,
    so the reconstruction settles before the KL and adversarial pressure kick in.

    Args:
        train_loader: Loader yielding (x_batch, s_batch, _) tuples for training.
        val_loader: Same, for validation.
        input_dim: Number of input features.
        attribute_types: One entry per sensitive attribute, 'binary' or 'regression'.
        latent_dim: Dimensionality of the latent space.
        num_epochs: Maximum number of epochs.
        lambda_adv: Weight of the adversarial term in the encoder loss.
        beta_vae: Weight of the KL divergence (the beta of a beta-VAE).
        lr: Learning rate of the encoder/decoder.
        lr_adv: Learning rate of the adversary.
        patience: Epochs without improvement before early stopping.
        class_weights: pos_weight of each attribute, offsetting class imbalance.
        _dataset_name: Dataset name, used in the plot and checkpoint paths.
        mitigation_type: Mitigation technique, used in the file names.
        model_type: Model tag recorded in the file names.
        optimizer: 'Adam' or 'SGD'.
        verbose: Whether to print progress and save plots/checkpoints.

    Returns:
        A tuple (encoder, decoder, adversary, best_val_loss), with the three
        models restored to the weights of the best validation epoch.
    """
    hidden_dim_vae  = int(input_dim // 1.2)
    hidden_dim_adv = int(latent_dim // 1.2)
    print(f"hidden_dim_vae: {hidden_dim_vae} and hidden_dim_adv: {hidden_dim_adv}")

    encoder = vae.Encoder(input_dim, latent_dim, hidden_dim_vae)
    decoder = vae.Decoder(latent_dim, input_dim, hidden_dim_vae)
    adversary = vae.MixedAdversary(latent_dim, attribute_types, hidden_dim_adv)

    encoder.to(device)
    decoder.to(device)
    adversary.to(device)

    if optimizer == "Adam":
        # Optimizer of the autoencoder (encoder + decoder)
        optimizer_vae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
        # Separate optimizer for the adversary
        optimizer_adv = optim.Adam(adversary.parameters(), lr=lr_adv)
    elif optimizer == "SGD":
        optimizer_vae = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, momentum=0.9, weight_decay=0.001)
        optimizer_adv = optim.SGD(adversary.parameters(), lr=lr_adv, momentum=0.9, weight_decay=0.001)

    early_stopper = EarlyStopper(patience=patience, min_delta=1e-4)

    best_val_loss = float('inf')
    best_encoder = None
    best_decoder = None
    best_adv = None
    
    if verbose:
        print("\nTraining the debiased VAE (adversarial)...")

    hist_vae_val = []
    hist_adv_val = []
    hist_vae_train = []
    hist_adv_train = []
    
    warmup = 100  # epochs over which beta and lambda_adv ramp up to full strength

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        adversary.train()
        
        total_vae_loss_train = 0.0
        total_adv_loss_train = 0.0
        batches = 0

        # Dynamic warmup of the hyperparameters, to stabilize the start of training
        current_beta = beta_vae * min(1.0, epoch / warmup)
        current_lambda = lambda_adv * min(1.0, epoch / warmup)

        for x_batch, s_batch, _ in train_loader:
            x_batch, s_batch = x_batch.to(device), s_batch.to(device)

            # --- Phase 1: Train the Autoencoder (Encoder + Decoder) ---
            # The objective is twofold:
            # 1. Minimize the reconstruction loss and the KL divergence (the VAE loss).
            # 2. Maximize the adversary loss (fool it), which strips the sensitive
            #    variable 's' out of the latent layer 'z'.
            optimizer_vae.zero_grad()
            
            mean, log_var = encoder(x_batch)
            z = vae.reparameterize(mean, log_var)
            recon = decoder(z)
            
            # VAE loss
            recon_loss, kl_div = vae.vae_losses(recon, x_batch, mean, log_var, recon_type="mse")
            vae_loss = recon_loss + current_beta * kl_div

            # Loss used to "fool" the adversary.
            # The gradient flows from the adversary back into the encoder.
            adv_preds_for_encoder = adversary(z)
            adv_loss_for_encoder = torch.tensor(0.0, device=device)
            for i, pred in enumerate(adv_preds_for_encoder):
                target = s_batch[:, i].unsqueeze(1) if s_batch.dim() > 1 else s_batch.unsqueeze(1)
                if attribute_types[i] == 'binary':
                    adv_loss_for_encoder += F.binary_cross_entropy_with_logits(
                        pred, target, reduction='mean', pos_weight=class_weights[i] if class_weights else None
                    )
                else: # Assuming 'regression' or similar
                    adv_loss_for_encoder += F.mse_loss(pred, target, reduction='mean')

            ## The total VAE loss adds the reconstruction/KL loss to the adversarial
            ## loss with a flipped sign. The encoder minimizes this total, which
            ## means it maximizes the adversary loss.
            total_loss_vae_step = vae_loss - current_lambda * adv_loss_for_encoder
            total_loss_vae_step.backward()
            optimizer_vae.step()

            # --- Phase 2: Train the Adversary ---
            # The goal is to make the adversary as good as possible at predicting
            # 's' from 'z'. Gradients must NOT flow back into the encoder.
            optimizer_adv.zero_grad()
            
            # Recompute z and detach it from the graph, so the encoder is untouched.
            with torch.no_grad():
                mean, log_var = encoder(x_batch)
                z_detached = vae.reparameterize(mean, log_var).detach()
            
            adv_preds = adversary(z_detached)
            adv_loss = 0.0
            for i, pred in enumerate(adv_preds):
                target = s_batch[:, i].unsqueeze(1) if s_batch.dim() > 1 else s_batch.unsqueeze(1)
                if attribute_types[i] == 'binary':
                    adv_loss += F.binary_cross_entropy_with_logits(
                        pred, target, reduction='mean', pos_weight=class_weights[i] if class_weights else None
                    )
                else: # Assuming 'regression' or similar
                    adv_loss += F.mse_loss(pred, target, reduction='mean')
            
            adv_loss.backward()
            optimizer_adv.step()

            total_vae_loss_train += vae_loss.item()
            total_adv_loss_train += adv_loss.item()
            batches += 1

        # --- Validation phase ---
        encoder.eval()
        decoder.eval()
        adversary.eval()
        
        total_vae_loss_val = 0.0
        total_adv_loss_val = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for x_val, s_val, _ in val_loader:
                x_val, s_val = x_val.to(device), s_val.to(device)
                
                mean, log_var = encoder(x_val)
                z = vae.reparameterize(mean, log_var)
                recon_val = decoder(z)
                
                # VAE validation loss (reconstruction + KL)
                recon_loss_val, kl_div_val = vae.vae_losses(recon_val, x_val, mean, log_var, recon_type="mse")
                vae_loss_val = recon_loss_val + current_beta * kl_div_val

                # Adversary validation loss
                adv_preds_val = adversary(z)
                adv_loss_val = 0
                for i, pred in enumerate(adv_preds_val):
                    target = s_val[:, i].unsqueeze(1) if s_val.dim() > 1 else s_val.unsqueeze(1)
                    if attribute_types[i] == 'binary':
                        adv_loss_val += F.binary_cross_entropy_with_logits(
                            pred, target, reduction='mean', pos_weight=class_weights[i] if class_weights else None)
                    else: # Consistent with the training loop
                        adv_loss_val += F.mse_loss(pred, target, reduction='mean')

                total_vae_loss_val += vae_loss_val.item()
                total_adv_loss_val += adv_loss_val.item()
                num_val_batches += 1

        # Average losses of the epoch
        avg_vae_loss_train = total_vae_loss_train / batches
        avg_adv_loss_train = total_adv_loss_train / batches
        avg_vae_loss_val = total_vae_loss_val / num_val_batches
        avg_adv_loss_val = total_adv_loss_val / num_val_batches

        hist_vae_train.append(avg_vae_loss_train)
        hist_adv_train.append(avg_adv_loss_train)
        hist_vae_val.append(avg_vae_loss_val)
        hist_adv_val.append(avg_adv_loss_val)

        ## Model selection tracks the VAE loss only: we want a model that
        ## reconstructs the data well, whatever the adversary scores. The adversary
        ## loss (avg_adv_loss_val) is merely monitored to confirm the debiasing is
        ## working, and is expected to rise or stay high.
        if avg_vae_loss_val < best_val_loss:
            best_val_loss = avg_vae_loss_val
            best_encoder = copy.deepcopy(encoder.state_dict())
            best_decoder = copy.deepcopy(decoder.state_dict())
            best_adv = copy.deepcopy(adversary.state_dict())
            # print(f"Epoch {epoch}: new best model saved with VAE Val Loss: {best_val_loss:.4f}")

        if early_stopper.early_stop(avg_vae_loss_val):
            print(f"Early stopping at epoch {epoch}")
            break
        
        if verbose and epoch % 5 == 0:
            print(f"[Ep {epoch}] Recon={recon_loss.item():.4f}, KL={kl_div.item():.4f}, AdvEnc={adv_loss_for_encoder.item():.4f}")


        if epoch % 10 == 0 or epoch == num_epochs - 1 and verbose:
                print("-" * 50)
                print(f"Epoch {epoch}:")
                print(f"  [Train] VAE Loss: {avg_vae_loss_train:.4f} | Adv Loss: {avg_adv_loss_train:.4f}")
                print(f"  [Val]    VAE Loss: {avg_vae_loss_val:.4f} | Adv Loss: {avg_adv_loss_val:.4f}")
                
                val_metrics = eval_adversary(encoder, adversary, val_loader, attribute_types, device, model_type)
                for k, v in val_metrics.items():
                    print(f"  [Val Adv Metrics] {k}: {v:.3f}")
                print("-" * 50)

    # Restore the best models found during training
    encoder.load_state_dict(best_encoder)
    decoder.load_state_dict(best_decoder)
    adversary.load_state_dict(best_adv)

    if verbose:
        # Plot the loss history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(hist_vae_train, label='VAE Loss (Train)')
        plt.plot(hist_vae_val, label='VAE Loss (Val)')
        plt.title('VAE Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(hist_adv_train, label='Adv Loss (Train)')
        plt.plot(hist_adv_val, label='Adv Loss (Val)')
        plt.title('Adversary Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        os.makedirs(f'./plots/{_dataset_name}', exist_ok=True)
        plt.savefig(f"./plots/{_dataset_name}/{_dataset_name}_{model_type}_{mitigation_type}_loss.pdf")

        # Save the model checkpoints
        os.makedirs(f'./model/{_dataset_name}', exist_ok=True)
        torch.save(encoder.state_dict(), f'./model/{_dataset_name}/trained_debiased_encoder_{_dataset_name}_{mitigation_type}.pth')
        torch.save(decoder.state_dict(), f'./model/{_dataset_name}/trained_debiased_decoder_{_dataset_name}_{mitigation_type}.pth')
        torch.save(adversary.state_dict(), f'./model/{_dataset_name}/trained_debiased_adversary_{_dataset_name}_{mitigation_type}.pth')

    return encoder, decoder, adversary, best_val_loss

def train_debiased_autoencoder(
    train_loader, val_loader, input_dim, attribute_types,
    latent_dim=32, num_epochs=100, lambda_adv=1e-3, lr=1e-3, lr_adv=1e-3,
    patience=10, class_weights=None, warmup=50, _dataset_name="dataset", mitigation_type="None", 
    model_type="ae", optimizer="Adam",
    verbose=True
):
    """
    Train a deterministic autoencoder adversarially, to debias its latent space.

    The deterministic counterpart of train_debiased_vae: the encoder emits z
    directly, so there is no reparameterization and no KL term. Only the
    reconstruction loss is traded against the adversary loss.

    Args:
        train_loader: Loader yielding (x_batch, s_batch, _) tuples for training.
        val_loader: Same, for validation.
        input_dim: Number of input features.
        attribute_types: One entry per sensitive attribute, 'binary' or 'regression'.
        latent_dim: Dimensionality of the latent space.
        num_epochs: Maximum number of epochs.
        lambda_adv: Weight of the adversarial term in the encoder loss.
        lr: Learning rate of the encoder/decoder.
        lr_adv: Learning rate of the adversary.
        patience: Epochs without improvement before early stopping.
        class_weights: pos_weight of each attribute, offsetting class imbalance.
        warmup: Epochs over which lambda_adv ramps up to full strength.
        _dataset_name: Dataset name, used in the plot and checkpoint paths.
        mitigation_type: Mitigation technique, used in the file names.
        model_type: Model tag recorded in the file names.
        optimizer: 'Adam' or 'SGD'.
        verbose: Whether to print progress and save plots/checkpoints.

    Returns:
        A tuple (encoder, decoder, adversary, best_val_loss), with the three
        models restored to the weights of the best validation epoch.
    """
    encoder = vae.Encoder(input_dim, latent_dim).to(device)
    decoder = vae.Decoder(latent_dim, input_dim).to(device)
    adversary = vae.MixedAdversary(latent_dim, attribute_types).to(device)

    if optimizer == "Adam":
        # Optimizer of the autoencoder (encoder + decoder)
        optimizer_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
        # Separate optimizer for the adversary: the two are trained apart
        optimizer_adv = optim.Adam(adversary.parameters(), lr=lr_adv)
    elif optimizer == "SGD":
        optimizer_ae = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, momentum=0.9, weight_decay=0.001)
        optimizer_adv = optim.SGD(adversary.parameters(), lr=lr_adv, momentum=0.9, weight_decay=0.001)

   
    best_val_loss = float('inf')
    best_encoder, best_decoder, best_adv = None, None, None

    hist_ae_train, hist_adv_train, hist_ae_val, hist_adv_val = [], [], [], []

    early_stopper = EarlyStopper(patience=patience, min_delta=1e-4)

    if verbose:
        print("\nTraining the debiased Autoencoder (adversarial)...")

    for epoch in range(num_epochs):
        encoder.train(); decoder.train(); adversary.train()
        total_ae_loss_train, total_adv_loss_train = 0, 0
        batches = 0

        # Dynamic warmup of lambda_adv
        current_lambda = lambda_adv * min(1.0, epoch / warmup)

        for x_batch, s_batch, _ in train_loader:
            x_batch, s_batch = x_batch.to(device), s_batch.to(device)

            # ---- Phase 1: Train the Autoencoder ----
            optimizer_ae.zero_grad()
            z = encoder(x_batch)
            recon = decoder(z)
            recon_loss = F.mse_loss(recon, x_batch, reduction='mean')

            # Adversarial loss seen by the encoder: subtracted, so minimizing
            # the total means maximizing the adversary loss
            adv_preds = adversary(z)
            adv_loss_enc = 0
            for i, pred in enumerate(adv_preds):
                target = s_batch[:, i].unsqueeze(1).float()
                if attribute_types[i] == 'binary':
                    adv_loss_enc += F.binary_cross_entropy_with_logits(
                        pred, target, reduction='mean',
                        pos_weight=torch.as_tensor(
                            class_weights[i], device=device, dtype=pred.dtype
                        ) if class_weights else None
                    )
                else:
                    adv_loss_enc += F.mse_loss(pred, target, reduction='mean')

            total_loss_ae = recon_loss - current_lambda * adv_loss_enc
            total_loss_ae.backward()
            optimizer_ae.step()

            # ---- Phase 2: Train the Adversary ----
            # z is detached, so these gradients never reach the encoder
            optimizer_adv.zero_grad()
            with torch.no_grad():
                z_det = encoder(x_batch).detach()
            adv_preds = adversary(z_det)
            adv_loss = 0
            for i, pred in enumerate(adv_preds):
                target = s_batch[:, i].unsqueeze(1).float()
                if attribute_types[i] == 'binary':
                    adv_loss += F.binary_cross_entropy_with_logits(
                        pred, target, reduction='mean',
                        pos_weight=torch.as_tensor(
                            class_weights[i], device=device, dtype=pred.dtype
                        ) if class_weights else None
                    )
                else:
                    adv_loss += F.mse_loss(pred, target, reduction='mean')

            adv_loss.backward()
            optimizer_adv.step()

            total_ae_loss_train += recon_loss.item()
            total_adv_loss_train += adv_loss.item()
            batches += 1

        # ---- Validation ----
        encoder.eval(); decoder.eval(); adversary.eval()
        total_ae_loss_val, total_adv_loss_val, num_val_batches = 0, 0, 0
        with torch.no_grad():
            for x_val, s_val, _ in val_loader:
                x_val, s_val = x_val.to(device), s_val.to(device)
                z = encoder(x_val)
                recon_val = decoder(z)
                recon_loss_val = F.mse_loss(recon_val, x_val, reduction='mean')

                adv_preds_val = adversary(z)
                adv_loss_val = 0
                for i, pred in enumerate(adv_preds_val):
                    target = s_val[:, i].unsqueeze(1).float()
                    if attribute_types[i] == 'binary':
                        adv_loss_val += F.binary_cross_entropy_with_logits(
                            pred, target, reduction='mean',
                            pos_weight=torch.as_tensor(
                                class_weights[i], device=device, dtype=pred.dtype
                            ) if class_weights else None
                        )
                    else:
                        adv_loss_val += F.mse_loss(pred, target, reduction='mean')

                total_ae_loss_val += recon_loss_val.item()
                total_adv_loss_val += adv_loss_val.item()
                num_val_batches += 1

        avg_ae_loss_train = total_ae_loss_train / batches
        avg_adv_loss_train = total_adv_loss_train / batches
        avg_ae_loss_val = total_ae_loss_val / num_val_batches
        avg_adv_loss_val = total_adv_loss_val / num_val_batches

        hist_ae_train.append(avg_ae_loss_train)
        hist_adv_train.append(avg_adv_loss_train)
        hist_ae_val.append(avg_ae_loss_val)
        hist_adv_val.append(avg_adv_loss_val)

        # Model selection tracks the reconstruction loss only
        if avg_ae_loss_val < best_val_loss:
            best_val_loss = avg_ae_loss_val
            best_encoder = copy.deepcopy(encoder.state_dict())
            best_decoder = copy.deepcopy(decoder.state_dict())
            best_adv = copy.deepcopy(adversary.state_dict())

        if early_stopper.early_stop(avg_ae_loss_val):
            print(f"Early stopping at epoch {epoch}")
            break

        if verbose and epoch % 5 == 0:
            print(f"[Ep {epoch:03d}] Recon={avg_ae_loss_val:.4f} | AdvVal={avg_adv_loss_val:.4f}")
            val_metrics = eval_adversary(encoder, adversary, val_loader, attribute_types, device, model_type)
            for k, v in val_metrics.items():
                print(f"  [Val Adv Metrics] {k}: {v:.3f}")
            print("-" * 50)

    # Restore the best weights found during training
    encoder.load_state_dict(best_encoder)
    decoder.load_state_dict(best_decoder)
    adversary.load_state_dict(best_adv)

    if verbose:
        # ---- Plot the history ----
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(hist_ae_train, label='AE Loss (Train)')
        plt.plot(hist_ae_val, label='AE Loss (Val)')
        plt.title('Autoencoder Loss History')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(hist_adv_train, label='Adv Loss (Train)')
        plt.plot(hist_adv_val, label='Adv Loss (Val)')
        plt.title('Adversary Loss History')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.tight_layout(); 
        os.makedirs(f'./plots/{_dataset_name}', exist_ok=True)
        plt.savefig(f"./plots/{_dataset_name}/{_dataset_name}_{model_type}_{mitigation_type}_loss.pdf")

        # ---- Save the checkpoints ----
        os.makedirs(f'./model/{_dataset_name}', exist_ok=True)
        torch.save(encoder.state_dict(), f'./model/{_dataset_name}/trained_debiasedAE_encoder_{_dataset_name}_{mitigation_type}.pth')
        torch.save(decoder.state_dict(), f'./model/{_dataset_name}/trained_debiasedAE_decoder_{_dataset_name}_{mitigation_type}.pth')
        torch.save(adversary.state_dict(), f'./model/{_dataset_name}/trained_debiasedAE_adversary_{_dataset_name}_{mitigation_type}.pth')

    return encoder, decoder, adversary, best_val_loss
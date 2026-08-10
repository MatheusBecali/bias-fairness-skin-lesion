# -*- coding: utf-8 -*-
"""
Autor: Matheus Becali Rocha
Email: matheusbecali@gmail.com
"""

import src.vae as vae
from utils.helpers import eval_adversary

import os
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Configuração do dispositivo
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
try:
    print(f"Dispositivo utilizado: {torch.cuda.get_device_name(device)}")
except Exception:
    print('Dispositivo CUDA não encontrado, utilizando CPU.')

_seed = 78645

class EarlyStopper:
    """
    Early stopping para interromper o treinamento quando uma métrica monitorada 
    para de melhorar.
    """
    def __init__(self, patience=10, min_delta=1e-4):
        """
        Args:
            patience (int): Quantas épocas esperar após a última vez que a 
                            métrica de validação melhorou.
            min_delta (float): Mudança mínima na métrica monitorada para ser 
                               considerada como uma melhoria.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop_flag = False

    def early_stop(self, val_loss):
        # Verifica se a perda de validação melhorou
        if val_loss < self.best_loss - self.min_delta:
            # Se melhorou, atualiza a melhor perda e reseta o contador
            self.best_loss = val_loss
            self.counter = 0
        else:
            # Se não melhorou, incrementa o contador
            self.counter += 1
            
        # Se o contador atingir o limite de paciência, ativa a flag de parada
        if self.counter >= self.patience:
            self.early_stop_flag = True
            
        return self.early_stop_flag

# Convert to Tensors and prepare data loaders
class ClassifyingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2, dropout_rate=0.1):
        super(ClassifyingNetwork, self).__init__()

        # self.MLPclassify = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.Dropout(dropout_rate),
        #     nn.ReLU(),
        #     # nn.ELU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.Dropout(dropout_rate),
        #     nn.ReLU(),
        #     # nn.ELU(),
        #     nn.Linear(hidden_size, num_classes)
        # )

        self.MLPclassify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Tanh(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = x.to(device)
        output = self.MLPclassify(x)
        return output

def train_debiased_vae(train_loader, val_loader, input_dim, attribute_types, latent_dim=100, 
                       num_epochs=100, lambda_adv=1e-3, beta_vae=1, lr=1e-3, lr_adv=1e-4, 
                       patience=10, class_weights=None, _dataset_name="dataset", mitigation_type="None", model_type="vae", 
                       optimizer="Adam", verbose=True):
 
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
        # Otimizador para o autoencoder (encoder + decoder)
        optimizer_vae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
        # Otimizador separado para o adversário
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
        print("\nTreinando VAE Desenviesado (Adversarial)...")

    hist_vae_val = []
    hist_adv_val = []
    hist_vae_train = []
    hist_adv_train = []
    
    warmup = 100

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        adversary.train()
        
        total_vae_loss_train = 0.0
        total_adv_loss_train = 0.0
        batches = 0

        # Aquecimento dinâmico dos hiperparâmetros para estabilizar o início do treino
        current_beta = beta_vae * min(1.0, epoch / warmup)
        current_lambda = lambda_adv * min(1.0, epoch / warmup)

        for x_batch, s_batch, _ in train_loader:
            x_batch, s_batch = x_batch.to(device), s_batch.to(device)

            # --- Fase 1: Treinar o Autoencoder (Encoder + Decoder) ---
            # O objetivo é duplo: 
            # 1. Minimizar a perda de reconstrução e o KL divergence (perda do VAE).
            # 2. Maximizar a perda do adversário (enganá-lo), o que remove a informação da variavel 
            #    sensivel 's' da camada latente 'z'.
            optimizer_vae.zero_grad()
            
            mean, log_var = encoder(x_batch)
            z = vae.reparameterize(mean, log_var)
            recon = decoder(z)
            
            # 1. Cálculo da perda do VAE
            
            # recon_loss = F.mse_loss(recon_batch, x_batch, reduction='mean')
            # kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            recon_loss, kl_div = vae.vae_losses(recon, x_batch, mean, log_var, recon_type="mse")
            vae_loss = recon_loss + current_beta * kl_div

            # 2. Cálculo da perda para "enganar" o adversário
            # O gradiente fluirá do adversário para o encoder.
            adv_preds_for_encoder = adversary(z)
            adv_loss_for_encoder = torch.tensor(0.0, device=device)
            for i, pred in enumerate(adv_preds_for_encoder):
                target = s_batch[:, i].unsqueeze(1) if s_batch.dim() > 1 else s_batch.unsqueeze(1)
                if attribute_types[i] == 'binary':
                    adv_loss_for_encoder += F.binary_cross_entropy_with_logits(
                        pred, target, reduction='mean', pos_weight=class_weights[i] if class_weights else None
                    )
                else: # Assumindo 'regression' ou similar
                    adv_loss_for_encoder += F.mse_loss(pred, target, reduction='mean')

            ## A perda total do VAE soma a perda de reconstrução/KL com a perda
            ## adversarial (com sinal invertido). O encoder tentará minimizar essa perda total,
            ## o que significa que ele irá maximizar a perda do adversário.
            total_loss_vae_step = vae_loss - current_lambda * adv_loss_for_encoder
            total_loss_vae_step.backward()
            optimizer_vae.step()

            # --- Fase 2: Treinar o Adversário ---
            # O objetivo é treinar o adversário para ser o melhor possível em prever 's' a partir de 'z'.
            # Os gradientes NÃO devem fluir para o encoder.
            optimizer_adv.zero_grad()
            
            # Re-calcula z e o destaca do grafo computacional para não treinar o encoder.
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
                else: # Assumindo 'regression' ou similar
                    adv_loss += F.mse_loss(pred, target, reduction='mean')
            
            adv_loss.backward()
            optimizer_adv.step()

            total_vae_loss_train += vae_loss.item()
            total_adv_loss_train += adv_loss.item()
            batches += 1

        # --- Fase de Validação ---
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
                
                # Perda de validação do VAE (reconstrução + KL)
                recon_loss_val, kl_div_val = vae.vae_losses(recon_val, x_val, mean, log_var, recon_type="mse")
                vae_loss_val = recon_loss_val + current_beta * kl_div_val

                # recon_loss_val = F.mse_loss(recon_val, x_val, reduction='mean')
                # kl_div_val = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                # kl_div_val     = torch.tensor(0.0, device=device)    # sem KL
                # vae_loss_val = recon_loss_val + current_beta * kl_div_val
                
                # Perda de validação do Adversário
                adv_preds_val = adversary(z)
                adv_loss_val = 0
                for i, pred in enumerate(adv_preds_val):
                    target = s_val[:, i].unsqueeze(1) if s_val.dim() > 1 else s_val.unsqueeze(1)
                    if attribute_types[i] == 'binary':
                        adv_loss_val += F.binary_cross_entropy_with_logits(
                            pred, target, reduction='mean', pos_weight=class_weights[i] if class_weights else None)
                    else: # ## CORREÇÃO/MELHORIA: Consistência com o loop de treino
                        adv_loss_val += F.mse_loss(pred, target, reduction='mean')

                total_vae_loss_val += vae_loss_val.item()
                total_adv_loss_val += adv_loss_val.item()
                num_val_batches += 1

        # Médias das perdas
        avg_vae_loss_train = total_vae_loss_train / batches
        avg_adv_loss_train = total_adv_loss_train / batches
        avg_vae_loss_val = total_vae_loss_val / num_val_batches
        avg_adv_loss_val = total_adv_loss_val / num_val_batches

        hist_vae_train.append(avg_vae_loss_train)
        hist_adv_train.append(avg_adv_loss_train)
        hist_vae_val.append(avg_vae_loss_val)
        hist_adv_val.append(avg_adv_loss_val)

        ## Queremos um modelo que reconstrua bem os dados, independentemente da performance do 
        ## adversário. A perda do adversário (avg_adv_loss_val) é monitorada para garantir que o
        ##  desenviesamento está funcionando (espera-se que ela suba ou se mantenha alta).
        if avg_vae_loss_val < best_val_loss:
            best_val_loss = avg_vae_loss_val
            best_encoder = copy.deepcopy(encoder.state_dict())
            best_decoder = copy.deepcopy(decoder.state_dict())
            best_adv = copy.deepcopy(adversary.state_dict())
            # print(f"Epoch {epoch}: Novo melhor modelo salvo com VAE Val Loss: {best_val_loss:.4f}")

        if early_stopper.early_stop(avg_vae_loss_val):
            print(f"Early stopping na época {epoch}")
            break
        
        if verbose:
            if epoch % 5 == 0:
                print(f"[Ép {epoch}] Recon={recon_loss.item():.4f}, KL={kl_div.item():.4f}, AdvEnc={adv_loss_for_encoder.item():.4f}")


        if epoch % 10 == 0 or epoch == num_epochs - 1:
            if verbose:
                print("-" * 50)
                print(f"Época {epoch}:")
                print(f"  [Treino] VAE Loss: {avg_vae_loss_train:.4f} | Adv Loss: {avg_adv_loss_train:.4f}")
                print(f"  [Val]    VAE Loss: {avg_vae_loss_val:.4f} | Adv Loss: {avg_adv_loss_val:.4f}")
                
                val_metrics = eval_adversary(encoder, adversary, val_loader, attribute_types, device, model_type)
                for k, v in val_metrics.items():
                    print(f"  [Val Métricas Adv] {k}: {v:.3f}")
                print("-" * 50)

    # Carregar os melhores modelos encontrados
    encoder.load_state_dict(best_encoder)
    decoder.load_state_dict(best_decoder)
    adversary.load_state_dict(best_adv)

    if verbose:
        # Plotar histórico de perdas
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(hist_vae_train, label='VAE Loss (Treino)')
        plt.plot(hist_vae_val, label='VAE Loss (Val)')
        plt.title('Histórico de Perda do VAE')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(hist_adv_train, label='Adv Loss (Treino)')
        plt.plot(hist_adv_val, label='Adv Loss (Val)')
        plt.title('Histórico de Perda do Adversário')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        os.makedirs(f'./plots/{_dataset_name}', exist_ok=True)
        plt.savefig(f"./plots/{_dataset_name}/{_dataset_name}_{model_type}_{mitigation_type}_loss.pdf")

        # Salvar modelos
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
    encoder = vae.Encoder(input_dim, latent_dim).to(device)
    decoder = vae.Decoder(latent_dim, input_dim).to(device)
    adversary = vae.MixedAdversary(latent_dim, attribute_types).to(device)

    if optimizer == "Adam":
        # Otimizador para o autoencoder (encoder + decoder)
        optimizer_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
        # Otimizador separado para o adversário
        optimizer_adv = optim.Adam(adversary.parameters(), lr=lr_adv)
    elif optimizer == "SGD":
        optimizer_ae = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, momentum=0.9, weight_decay=0.001)
        optimizer_adv = optim.SGD(adversary.parameters(), lr=lr_adv, momentum=0.9, weight_decay=0.001)

   
    best_val_loss = float('inf')
    best_encoder, best_decoder, best_adv = None, None, None

    hist_ae_train, hist_adv_train, hist_ae_val, hist_adv_val = [], [], [], []

    early_stopper = EarlyStopper(patience=patience, min_delta=1e-4)

    if verbose:
        print("\nTreinando Autoencoder Desenviesado (Adversarial)...")

    for epoch in range(num_epochs):
        encoder.train(); decoder.train(); adversary.train()
        total_ae_loss_train, total_adv_loss_train = 0, 0
        batches = 0

        # Warmup dinâmico de λ_adv
        current_lambda = lambda_adv * min(1.0, epoch / warmup)

        for x_batch, s_batch, _ in train_loader:
            x_batch, s_batch = x_batch.to(device), s_batch.to(device)

            # ---- Fase 1: Treinar Autoencoder ----
            optimizer_ae.zero_grad()
            z = encoder(x_batch)
            recon = decoder(z)
            recon_loss = F.mse_loss(recon, x_batch, reduction='mean')

            # Perda adversarial para o encoder (inverter sinal)
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

            # ---- Fase 2: Treinar Adversário ----
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

        # ---- Validação ----
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

        if avg_ae_loss_val < best_val_loss:
            best_val_loss = avg_ae_loss_val
            best_encoder = copy.deepcopy(encoder.state_dict())
            best_decoder = copy.deepcopy(decoder.state_dict())
            best_adv = copy.deepcopy(adversary.state_dict())

        if early_stopper.early_stop(avg_ae_loss_val):
            print(f"Early stopping na época {epoch}")
            break

        if verbose and epoch % 5 == 0:
            print(f"[Ép {epoch:03d}] Recon={avg_ae_loss_val:.4f} | AdvVal={avg_adv_loss_val:.4f}")
            val_metrics = eval_adversary(encoder, adversary, val_loader, attribute_types, device, model_type)
            for k, v in val_metrics.items():
                print(f"  [Val Métricas Adv] {k}: {v:.3f}")
            print("-" * 50)

    # Carregar melhores pesos
    encoder.load_state_dict(best_encoder)
    decoder.load_state_dict(best_decoder)
    adversary.load_state_dict(best_adv)

    if verbose:
        # ---- Plotar histórico ----
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(hist_ae_train, label='AE Loss (Treino)')
        plt.plot(hist_ae_val, label='AE Loss (Val)')
        plt.title('Histórico de Perda do Autoencoder')
        plt.xlabel('Época'); plt.ylabel('Perda'); plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(hist_adv_train, label='Adv Loss (Treino)')
        plt.plot(hist_adv_val, label='Adv Loss (Val)')
        plt.title('Histórico de Perda do Adversário')
        plt.xlabel('Época'); plt.ylabel('Perda'); plt.legend()
        plt.tight_layout(); 
        os.makedirs(f'./plots/{_dataset_name}', exist_ok=True)
        plt.savefig(f"./plots/{_dataset_name}/{_dataset_name}_{model_type}_{mitigation_type}_loss.pdf")

        # ---- Salvar ----
        os.makedirs(f'./model/{_dataset_name}', exist_ok=True)
        torch.save(encoder.state_dict(), f'./model/{_dataset_name}/trained_debiasedAE_encoder_{_dataset_name}_{mitigation_type}.pth')
        torch.save(decoder.state_dict(), f'./model/{_dataset_name}/trained_debiasedAE_decoder_{_dataset_name}_{mitigation_type}.pth')
        torch.save(adversary.state_dict(), f'./model/{_dataset_name}/trained_debiasedAE_adversary_{_dataset_name}_{mitigation_type}.pth')

    return encoder, decoder, adversary, best_val_loss
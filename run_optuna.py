# -*- coding: utf-8 -*-
"""
Autor: Matheus Becali Rocha
Email: matheusbecali@gmail.com
"""


# ================================================================================================ #
# Libraries
# ================================================================================================ #
import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import optuna
import argparse

from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                           precision_score, recall_score)
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight

from collections import defaultdict

# for multiclass
from holisticai.bias.metrics import (multiclass_statistical_parity, multiclass_true_rates, 
                                     multiclass_equality_of_opp, multiclass_average_odds)

# for binaryclass
from holisticai.bias.metrics import (average_odds_diff, disparate_impact, equal_opportunity_diff,
                                     statistical_parity)
# Add 06/08/2025 - Post-Processing
from holisticai.bias.mitigation import (CalibratedEqualizedOdds, 
                                        LPDebiaserMulticlass, 
                                        MLDebiaser)

from src.net import train_debiased_vae, ClassifyingNetwork, train_debiased_autoencoder
from utils.helpers import (features_setting, preprocess_dataset,
                           calculate_class_weights, prepare_data_loader, evaluate_fairness_latent, 
                           save_results_to_csv)
from src.vae import reparameterize

# ================================================================================================ #

# Configuração do dispositivo
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
try:
    print(f"Dispositivo utilizado: {torch.cuda.get_device_name(device)}")
except Exception:
    print('Dispositivo CUDA não encontrado, utilizando CPU.')

_seed = 78645


def build_split_iterator(X_cv, stratify_cv, k_folds, fixed_validation_mask=None):
    """
    Retorna iterador de splits.
    Se fixed_validation_mask for informado, usa somente 1 split fixo (treino/validação por fold).
    """
    if fixed_validation_mask is not None:
        mask = np.array(fixed_validation_mask, dtype=bool)
        train_idx = np.where(~mask)[0]
        valid_idx = np.where(mask)[0]

        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise ValueError(
                "Split por fold inválido: conjunto de treino ou validação ficou vazio."
            )

        return [(train_idx, valid_idx)], 1

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=_seed)
    return list(kf.split(X_cv, stratify_cv)), k_folds

def generate_and_save_debiased_data_with_sensitive_info(encoder, decoder, dataloader, scaler, columns, 
                                                        label_columns, sensitive_columns, filename, model_type="vae", verbose=False):
    encoder.eval()
    decoder.eval()
    debiased_outputs, labels_outputs, sensitive_outputs = [], [], []

    with torch.no_grad():
        for X_batch, s_batch, y_batch in dataloader:  # 's_batch' são os dados sensíveis
            X_batch = X_batch.to(device)
            s_batch = s_batch.to(device)
            y_batch = y_batch.to(device)

            if model_type == "vae":
                mean, log_var = encoder(X_batch)
                # z = reparameterize(mean, log_var)
                recon_batch = decoder(mean)
                # recon_batch = decoder(mean)
            elif model_type == "ae":
                z = encoder(X_batch)
                recon_batch = decoder(z)
            else:
                raise ValueError("model_type deve ser 'vae' ou 'ae'")
                
            debiased_outputs.append(recon_batch.cpu().numpy())
            labels_outputs.append(y_batch.cpu().numpy())
            sensitive_outputs.append(s_batch.cpu().numpy())

    X_debiased_scaled = np.concatenate(debiased_outputs, axis=0)

    X_debiased_original = X_debiased_scaled.copy()
    
    # Aplica inverse_transform APENAS nas colunas não-binárias
    # X_debiased_original[:, indices_nao_bin] = scaler.inverse_transform(
    #     X_debiased_scaled[:, indices_nao_bin]
    # )

    X_debiased_original = scaler.inverse_transform(X_debiased_scaled)
    
    y_full = np.concatenate(labels_outputs, axis=0)
    # Concatenando os dados sensíveis
    s_full = np.concatenate(sensitive_outputs, axis=0)

    # Criando o DataFrame com os dados desenviesados e os dados sensíveis
    df_debiased = pd.DataFrame(X_debiased_original, columns=columns)
    df_debiased[label_columns] = y_full

    # Adicionando as colunas sensíveis
    for i, sensitive_column in enumerate(sensitive_columns):
        df_debiased[sensitive_column] = s_full[:, i]

    # Salvando os dados no arquivo CSV
    df_debiased.to_csv(filename, index=False)

    if verbose:
        print(
            f"\nDados desenviesados e sensíveis salvos em '{filename}' (na escala original).")

    return df_debiased

def train_mlp(_dataset_name, X_cv, y_cv, X_test_biased, y_test, stratify_cv, sensitive_features, 
              _set_loss, mitigation_tech, 
              opt_type="Adam", 
              batch_size=32, 
              k_folds=5, 
              _epochs=2000, 
              verbose=True, 
              lr=0.001,
              hidden_size_ratio=2,  # input_size // hidden_size_ratio
              weight_decay=0.001,
              sched_factor=0.1,
              sched_patience=20,
              fixed_validation_mask=None):
    """
    Treina uma MLP com validação cruzada estratificada.
    
    Novos parâmetros:
    - lr: Learning rate
    - hidden_size_ratio: Razão entre input_size e hidden_size (ex: 2 → hidden_size = input_size // 2)
    - weight_decay: Regularização L2
    - dropout: Taxa de dropout (0 = sem dropout)
    - sched_factor: Fator de redução do scheduler
    - sched_patience: Paciência do scheduler
    """

    # --- ESTRUTURA PARA ARMAZENAR MÉTRICAS DETALHADAS POR FOLD ---
    if _dataset_name in ["db-pad-ufes-20", "db-hiba", "db-midas"]:
        sp, di, eod, aod = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    else:
        raise NotImplementedError(f"Invalid Dataset: {_dataset_name}")

    # Inicializar estratégia de split
    split_iterator, total_folds = build_split_iterator(
        X_cv, stratify_cv, k_folds, fixed_validation_mask=fixed_validation_mask
    )

    # Lista para armazenar métricas de cada fold
    metrics_list = []
    accuracy = []
    balancedAccuracyScore = []
    recall = []
    precision = []
    f1 = []
    auc = []
    test_loss = []

    # Loop de validação
    for fold, (train_idx, valid_idx) in enumerate(split_iterator):
        curr_loss = 0
        limit_stop = 20
        prev_loss = np.inf
        trigger_times = 0

        if verbose:
            print(f"\nFold {fold+1}/{total_folds}")

        # Criar DataLoaders para cada fold
        X_train_biased = X_cv.iloc[train_idx].reset_index(drop=True)
        y_train = y_cv.iloc[train_idx]
        X_valid_biased = X_cv.iloc[valid_idx].reset_index(drop=True)
        y_valid = y_cv.iloc[valid_idx]

        # Remover variável sensível
        X_train_no_sensitive = X_train_biased.drop(columns=sensitive_features)
        X_valid_no_sensitive = X_valid_biased.drop(columns=sensitive_features)
        X_test_no_sensitive = X_test_biased.drop(columns=sensitive_features)

        # Normalização
        scaler = StandardScaler()
        X_train_scaler_np = scaler.fit_transform(X_train_no_sensitive)
        X_valid_scaler_np = scaler.transform(X_valid_no_sensitive)
        X_test_scaler_np = scaler.transform(X_test_no_sensitive)

        X_train_scaler = pd.DataFrame(X_train_scaler_np, columns=X_train_no_sensitive.columns, index=X_train_no_sensitive.index)
        X_valid_scaler = pd.DataFrame(X_valid_scaler_np, columns=X_valid_no_sensitive.columns, index=X_valid_no_sensitive.index)
        X_test_scaler = pd.DataFrame(X_test_scaler_np, columns=X_test_no_sensitive.columns, index=X_test_no_sensitive.index)

        # Tensores
        X_train_tensor = torch.tensor(np.array(X_train_scaler), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
        X_valid_tensor = torch.tensor(np.array(X_valid_scaler), dtype=torch.float32)
        y_valid_tensor = torch.tensor(y_valid.to_numpy(), dtype=torch.long)
        X_test_tensor = torch.tensor(np.array(X_test_scaler), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

        # Class weights
        y_train_np = y_train_tensor.cpu().numpy()
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_np), y=y_train_np)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        # DataLoaders
        train_dataloader, _ = prepare_data_loader(
            X_train_tensor,
            y_train_tensor,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        valid_dataloader, _ = prepare_data_loader(X_valid_tensor, y_valid_tensor, batch_size=batch_size, shuffle=False)
        test_dataloader, _ = prepare_data_loader(X_test_tensor, y_test_tensor, batch_size=batch_size, shuffle=False)

        # Configurações do modelo
        _input_size = X_train_scaler.shape[1]
        _hidden_size = _input_size // hidden_size_ratio  # NOVO: parametrizável
        _num_classes = len(np.unique(y_train))

        # Criar modelo com dropout (se necessário, você precisa modificar ClassifyingNetwork)
        model = ClassifyingNetwork(
            input_size=_input_size,
            hidden_size=_hidden_size,
            num_classes=_num_classes
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer com learning rate e weight decay customizáveis
        if opt_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"Invalid Optimizer: {opt_type}")

        # Scheduler com parâmetros customizáveis
        scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=sched_factor,
            min_lr=1e-6,
            patience=sched_patience
        )

        # Training loop
        for epoch in range(_epochs):
            model.train()
            train_total_loss, valid_total_loss = 0.0, 0.0
            y_true_train, y_pred_train = [], []
            y_true_valid, y_pred_valid = [], []

            for X_batch, y_batch in train_dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                tr_loss = criterion(outputs, y_batch)
                tr_loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                y_true_train.extend(y_batch.tolist())
                y_pred_train.extend(predicted.cpu().tolist())
                train_total_loss += tr_loss.item()

            train_bacc = balanced_accuracy_score(y_true_train, y_pred_train)
            train_epoch_loss = train_total_loss / len(train_dataloader)

            if verbose:
                print(f'Epoch [{epoch+1}/{_epochs}]')
                print(f"Train -> Loss: {train_epoch_loss:.4f}, BACC: {train_bacc:.4f}")

            # Validação
            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in valid_dataloader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs.data, 1)

                    v_loss = criterion(outputs, y_batch)
                    valid_total_loss += v_loss.item()

                    y_true_valid.extend(y_batch.tolist())
                    y_pred_valid.extend(predicted.cpu().tolist())

                valid_bacc = balanced_accuracy_score(y_true_valid, y_pred_valid)
                valid_epoch_loss = valid_total_loss / len(valid_dataloader)

                scheduler_lr.step(valid_epoch_loss)
                if verbose:
                    print("Current LR:", scheduler_lr.get_last_lr())
                    print(f"Valid -> Loss: {valid_epoch_loss:.4f}, BACC: {valid_bacc:.4f}")
                curr_loss = valid_epoch_loss

            # Early stopping
            if curr_loss > prev_loss:
                trigger_times += 1
                if verbose:
                    print(f"prev_loss: {prev_loss}, curr_loss: {curr_loss}")
                    print(f'Times without improved: {trigger_times}')

                if trigger_times >= limit_stop:
                    if verbose:
                        print(f'[*] Early stopping in Epoch: {epoch}!')
                    break
            else:
                trigger_times = 0
                prev_loss = curr_loss

        # Avaliação no conjunto de Validação (para possível pós-processamento)
        model.eval()
        y_true_valid_pp, y_proba_valid = [], []
        with torch.no_grad():
            for X_batch, y_batch in valid_dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                probs = F.softmax(outputs, dim=1)
                y_true_valid_pp.extend(y_batch.tolist())
                y_proba_valid.extend(probs.cpu().numpy())

        # Avaliação no conjunto de Teste
        test_total_loss = 0.0
        with torch.no_grad():
            y_true_test, y_pred_test, y_proba_test = [], [], []
            for X_batch, y_batch in test_dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)

                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                te_loss = criterion(outputs, y_batch)
                test_total_loss += te_loss.item()

                y_true_test.extend(y_batch.tolist())
                y_pred_test.extend(predicted.cpu().tolist())
                y_proba_test.extend(probs.cpu().numpy())

        test_epoch_loss = test_total_loss / len(test_dataloader)

        if _dataset_name in ["db-pad-ufes-20", "db-hiba", "db-midas"]:
            sp_fold, di_fold, eod_fold, aod_fold = {}, {}, {}, {}

        accuracy_fold, balancedAccuracyScore_fold = {}, {}
        precision_fold, recall_fold, f1_fold = {}, {}, {}
        
        # Preserva as predições originais antes do loop de atributos sensíveis
        y_pred_test_original = list(y_pred_test)
        y_proba_test_original = list(y_proba_test)

        # Calcule as métricas de fairness para TODOS os atributos, independentemente da estratégia
        for sens_attr in sensitive_features:
            # Reseta predições para cada atributo sensível
            y_pred_test = list(y_pred_test_original)

            A_test = X_test_biased[sens_attr]

            # Define os dois grupos diretamente. 
            # Assumimos que o grupo "privilegiado" é 0 e o "desprivilegiado" é 1.
            # Adapte se a sua codificação for diferente.
            group_a_test = (A_test == 1)  # Grupo desprivilegiado (ex: 'outros', 'mulher')
            group_b_test = (A_test == 0)  # Grupo privilegiado (ex: 'branco', 'homem')

            if mitigation_tech in ["Pos", "PP", "IP", "PIP"]:
                if verbose:
                    if mitigation_tech == "PIP":
                        print(f"Rodando com tecnica de mitigação: {mitigation_tech} - Etapa 3: Pos-processamento!")
                    elif mitigation_tech in ["PP", "IP"]:
                        print(f"Rodando com tecnica de mitigação: {mitigation_tech} - Etapa 2: Pos-processamento!")
                    else:
                        print(f"Rodando com tecnica de mitigação: {mitigation_tech}!")
                    
                # Evita data leakage: ajusta o mitigador na validação e aplica no teste.
                A_valid = X_valid_biased[sens_attr]
                group_a_valid = (A_valid == 1)
                group_b_valid = (A_valid == 0)

                mitigator = MLDebiaser(gamma=0.1)
                mitigator.fit(y_proba_valid, group_a=group_a_valid, group_b=group_b_valid)
                y_pred_test_cpp = mitigator.transform(
                    y_proba_test_original,
                    group_a=group_a_test,
                    group_b=group_b_test,
                )['y_pred']

                # Use y_pred_test_cpp para fairness deste atributo
                y_pred_test = y_pred_test_cpp
            else:
                pass

            # Calcula performance para este atributo sensível (predições podem mudar no pós-processamento).
            accuracy_fold[sens_attr] = accuracy_score(y_true_test, y_pred_test)
            balancedAccuracyScore_fold[sens_attr] = balanced_accuracy_score(y_true_test, y_pred_test)
            recall_fold[sens_attr] = recall_score(y_true_test, y_pred_test, average='weighted')
            precision_fold[sens_attr] = precision_score(
                y_true_test,
                y_pred_test,
                average='weighted',
                zero_division=True,
            )
            f1_fold[sens_attr] = f1_score(y_true_test, y_pred_test, average='weighted')

            if np.array_equal(np.unique(y_true_test), [0, 1]):
                # Calcula a disparidade única entre os dois grupos
                sp_value = statistical_parity(group_a_test, group_b_test, y_pred_test)
                di_value = disparate_impact(group_a_test, group_b_test, y_pred_test)
                eod_value = equal_opportunity_diff(group_a_test, group_b_test, y_pred_test, y_true_test)
                aod_value = average_odds_diff(group_a_test, group_b_test, y_pred_test, y_true_test)


                # Para salvar no csv de folders
                sp_fold[sens_attr] = np.abs(sp_value)
                di_fold[sens_attr] = np.abs(di_value)
                eod_fold[sens_attr] = np.abs(eod_value)
                aod_fold[sens_attr] = np.abs(aod_value)

                # Adiciona o valor absoluto da disparidade diretamente à lista do atributo
                sp[sens_attr].append(np.abs(sp_value))
                di[sens_attr].append(np.abs(di_value))
                eod[sens_attr].append(np.abs(eod_value))
                aod[sens_attr].append(np.abs(aod_value))
            
            else:
                pass
            


        # Agrega as métricas por fold considerando todos os atributos sensíveis.
        accuracy_fold_mean = float(np.mean([accuracy_fold[s] for s in sensitive_features if s in accuracy_fold])) if len(accuracy_fold) > 0 else np.nan
        bacc_fold_mean = float(np.mean([balancedAccuracyScore_fold[s] for s in sensitive_features if s in balancedAccuracyScore_fold])) if len(balancedAccuracyScore_fold) > 0 else np.nan
        recall_fold_mean = float(np.mean([recall_fold[s] for s in sensitive_features if s in recall_fold])) if len(recall_fold) > 0 else np.nan
        precision_fold_mean = float(np.mean([precision_fold[s] for s in sensitive_features if s in precision_fold])) if len(precision_fold) > 0 else np.nan
        f1_fold_mean = float(np.mean([f1_fold[s] for s in sensitive_features if s in f1_fold])) if len(f1_fold) > 0 else np.nan

        if not np.isnan(bacc_fold_mean):
            test_loss.append(test_epoch_loss)
            accuracy.append(accuracy_fold_mean)
            balancedAccuracyScore.append(bacc_fold_mean)
            recall.append(recall_fold_mean)
            precision.append(precision_fold_mean)
            f1.append(f1_fold_mean)

    return float(np.mean(balancedAccuracyScore)) if len(balancedAccuracyScore) > 0 else 0.0

def optuna_run(_dataset_name):
    """
    Otimização de hiperparâmetros com Optuna para pipeline VAE + MLP
    Adaptado para a implementação atual do train_mlp
    """
    _mitigation_tech = "PIP"
    _seed = 78645
    _n_trials = 200
    _num_epochs_vae = 200  # AUMENTADO de 1 para 50
    _early_stop_patience = 20
    _verbose = False
    _type_adv = "vae"


    if _dataset_name in ["db-pad-ufes-20"]:
        _attribute_types = ['binary', 'binary'] # gender, fitz
    elif _dataset_name in ["db-hiba", "db-midas"]:
        _attribute_types = ['binary'] # gender
    else:
        raise NotImplementedError(f"Invalid Dataset: {_dataset_name}")

    # args para salvar os dados desenviesados
    _path = f"./debiased/{_dataset_name}/dados_desenviesados_com_sensitivos"
    _filename=f"{_path}_{_mitigation_tech}_{_dataset_name}.csv"
    os.makedirs(os.path.dirname(f"./debiased/{_dataset_name}"), exist_ok=True)
    os.makedirs(os.path.dirname(_path), exist_ok=True)

    # args para a MLP
    _k_folds = 5
    _epochs = 1
    _set_loss = "weighted_cross_entropy_loss"
    _validation_fold = 1  # Fold fixo para validação (consistente com main.py)
    ######################################################################################

    # Funções de conversão (mesmas do main.py)
    def convert_fitzpatrick_scale(df):
        if 'fitzpatrick' in df.columns:
            df['fitzpatrick'] = df['fitzpatrick'].apply(lambda x: 0 if x < 3 else 1)
        return df

    def convert_diagnosis(df):
        if 'diagnosis' in df.columns:
            df['diagnosis'] = df['diagnosis'].apply(lambda x: 0 if x == "NC" or x == "benign" else 1)
        return df

    # Read datasets (treino+val e teste separados, como no main.py)
    df_data = pd.read_csv(f'./data/{_dataset_name}/processed_{_dataset_name}.csv', delimiter=',')
    df_data_test = pd.read_csv(f'./data/{_dataset_name}/processed_{_dataset_name}_test.csv', delimiter=',')

    if _dataset_name == "db-midas":
        # Limpa e padroniza nomes de colunas no DataFrame
        df_data.columns = (
            df_data.columns
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.lower()
        )
        df_data_test.columns = (
            df_data_test.columns
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.lower()
        )

    # Aplicar conversões em ambos os datasets
    df_data = convert_fitzpatrick_scale(df_data)
    df_data_test = convert_fitzpatrick_scale(df_data_test)
    df_data = convert_diagnosis(df_data)
    df_data_test = convert_diagnosis(df_data_test)

    # Regra simplificada: as duas primeiras colunas são sempre não-preditivas.
    non_feature_columns = list(df_data.columns[:2])

    if "fold" not in df_data.columns:
        raise ValueError("A coluna 'fold' não foi encontrada em df_data.")

    """Setup features"""
    dict_ = features_setting(f"{_dataset_name}")
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    discrete_features = dict_["discrete_features"]
    full_features = dict_["full_features"]
    standard_features = continuous_features + discrete_features
    target = dict_["target"]

    df_data[target] = df_data[target].astype(float)
    df_data_test[target] = df_data_test[target].astype(float)

    # Split fixo por fold (consistente com main.py)
    train_mask = df_data["fold"].astype(int) != int(_validation_fold)
    val_mask = df_data["fold"].astype(int) == int(_validation_fold)

    df_train = df_data.loc[train_mask].copy()
    df_val = df_data.loc[val_mask].copy()

    # ==============================================================================
    # Rodando DEMV — SOMENTE nos dados de TREINO para evitar data leakage
    # ==============================================================================
    if _mitigation_tech in ["Pre", "PI", "PP", "PIP"]:
        if _verbose:
            print(f"Rodando DEMV (Pre-processamento) somente nos dados de treino!")

        from demv import DEMV

        demv = DEMV(sensitive_vars=sensitive_features, round_level=1, verbose=False)
        demv_x = df_train.drop(
            columns=[target] + [c for c in non_feature_columns if c in df_train.columns]
        )
        demv_y = df_train[target]
        x_new, y_new = demv.fit_transform(demv_x, demv_y)
        # Reconstrói df_train mantendo as colunas não-preditivas
        # O DEMV gera novas instâncias (oversampling), então o `x_new` é maior que o `df_train` original.
        df_train_new = x_new.copy()
        df_train_new[target] = y_new.copy()

        # Precisamos restaurar as colunas "non_feature" (img_id, fold) para NÃO quebrar lá na frente.
        # Para as linhas originais (as N primeiras):
        n_orig = len(df_train)
        for c in non_feature_columns:
            if c in df_train.columns:
                col_data = list(df_train[c])
                
                # Preenche os dados sintéticos gerados pelo DEMV com valores defaults de treino
                if c == "fold":
                    # Usa um fold fictício (-1) garantindo que continue classificado como não-validação
                    col_data += [-1] * (len(df_train_new) - n_orig)
                else:
                    col_data += ["synthetic"] * (len(df_train_new) - n_orig)
                    
                df_train_new[c] = col_data
        
        df_train = df_train_new.copy()

    feature_cols = [
        c for c in normal_features
        if c not in non_feature_columns and c in df_data.columns
    ]

    df_data_no_sensitive = df_data[normal_features]
    df_sensitive = df_data[sensitive_features]

    def objective(trial):
        """
        Função objetivo para otimização com Optuna
        """
        # =================================================================
        # HIPERPARÂMETROS DO VAE/AE
        # =================================================================
        if _dataset_name == "db-pad-ufes-20":
            lr_vae = trial.suggest_float('lr_vae', 1e-5, 1e-2, log=True)
            lr_adv = trial.suggest_float('lr_adv', 1e-6, 1e-3, log=True)
            lambda_adv = trial.suggest_float('lambda_adv', 1e-3, 1.0, log=True)
            beta = trial.suggest_float('beta', 1e-3, 1.0, log=True)
            batch_size_vae = trial.suggest_categorical("batch_size_vae", [8, 16, 32, 48])
            optimizer_vae = trial.suggest_categorical("optimizer_vae", ["Adam", "SGD"])
            
        elif _dataset_name in ["db-hiba", "db-midas"]:
            lr_vae = trial.suggest_float("lr_vae", 1e-5, 5e-3, log=True)
            lr_adv = trial.suggest_float("lr_adv", 1e-5, 1e-2, log=True)
            lambda_adv = trial.suggest_float("lambda_adv", 1e-2, 5.0, log=True)
            beta = trial.suggest_float('beta', 1e-2, 10.0, log=True)
            batch_size_vae = trial.suggest_categorical("batch_size_vae", [8, 16, 32, 48])
            optimizer_vae = trial.suggest_categorical("optimizer_vae", ["Adam", "SGD"])
        
        # =================================================================
        # HIPERPARÂMETROS DA MLP (limitados aos que sua função aceita)
        # =================================================================
        batch_size_mlp = trial.suggest_categorical("batch_size_mlp", [16, 32, 64, 128])
        optimizer_mlp = trial.suggest_categorical("optimizer_mlp", ["Adam", "SGD"])
        # epochs_mlp = trial.suggest_int('epochs_mlp', 100, 500)  # Aumentado de 2000
        epochs_mlp = 50
        
        # =================================================================
        # TREINAMENTO DO VAE/AE (In-Processing)
        # =================================================================
        if _mitigation_tech in ["In", "PI", "IP", "PIP"]:
            # Preparar dados para PyTorch usando fold split (consistente com main.py)
            X_train = np.array(df_train[feature_cols]).astype(np.float32)
            y_label_train = np.array(df_train[target]).astype(np.float32)
            y_sensitive_train = np.array(df_train[sensitive_features]).astype(np.float32)
            fold_train = df_train["fold"].astype(int).to_numpy() if "fold" in df_train.columns else np.array([])

            X_val = np.array(df_val[feature_cols]).astype(np.float32)
            y_label_val = np.array(df_val[target]).astype(np.float32)
            y_sensitive_val = np.array(df_val[sensitive_features]).astype(np.float32)
            fold_val = df_val["fold"].astype(int).to_numpy() if "fold" in df_val.columns else np.array([])

            # Dados de teste separados
            df_data_test_no_sensitive = df_data_test[feature_cols]
            df_data_test_sensitive = df_data_test[sensitive_features]
            X_test = np.array(df_data_test_no_sensitive).astype(np.float32)
            y_label_test = np.array(df_data_test[target]).astype(np.float32)
            y_sensitive_test = np.array(df_data_test_sensitive).astype(np.float32)
            fold_test = df_data_test["fold"].astype(int).to_numpy() if "fold" in df_data_test.columns else np.array([])
            
            # Normalização
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            
            # DataLoaders
            def get_dataloader(X, sensitive_attrs, labels, batch_size, shuffle):
                dataset = TensorDataset(
                    torch.from_numpy(X), 
                    torch.from_numpy(sensitive_attrs), 
                    torch.from_numpy(labels)
                )
                return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            
            dataloader_train = get_dataloader(X_train, y_sensitive_train, y_label_train, batch_size_vae, True)
            dataloader_val = get_dataloader(X_val, y_sensitive_val, y_label_val, batch_size_vae, False)
            dataloader_test = get_dataloader(X_test, y_sensitive_test, y_label_test, batch_size_vae, False)
            
            # Combinar train + val para dados finais
            X_data_normalized = np.concatenate([X_train, X_val], axis=0)
            y_sensitive_data = np.concatenate([y_sensitive_train, y_sensitive_val], axis=0)
            y_label_data = np.concatenate([y_label_train, y_label_val], axis=0)
            fold_data = np.concatenate([fold_train, fold_val], axis=0) if fold_train.size > 0 and fold_val.size > 0 else np.array([])
            dataloader_data = get_dataloader(X_data_normalized, y_sensitive_data, y_label_data, batch_size_vae, False)
            
            # Configurações do VAE
            latent_dims = int(X_train.shape[1] // 1.2)
            pos_weights = calculate_class_weights(y_sensitive_train, device, verbose=False)
            
            # Treinar VAE/AE
            try:
                if _type_adv == "vae":
                    encoder_debiased, decoder_debiased, mixed_adversary, vae_metrics = train_debiased_vae(
                        train_loader=dataloader_train,
                        val_loader=dataloader_val,
                        input_dim=X_train.shape[1],
                        attribute_types=_attribute_types,
                        latent_dim=latent_dims,
                        num_epochs=_num_epochs_vae,
                        patience=_early_stop_patience,
                        lambda_adv=lambda_adv,
                        beta_vae=beta,
                        lr_adv=lr_adv,
                        lr=lr_vae,
                        model_type=_type_adv,
                        class_weights=pos_weights,
                        optimizer=optimizer_vae,
                        verbose=False
                    )
                elif _type_adv == "ae":
                    encoder_debiased, decoder_debiased, mixed_adversary, vae_metrics = train_debiased_autoencoder(
                        train_loader=dataloader_train,
                        val_loader=dataloader_val,
                        input_dim=X_train.shape[1],
                        attribute_types=_attribute_types,
                        latent_dim=latent_dims,
                        num_epochs=_num_epochs_vae,
                        lambda_adv=lambda_adv,
                        lr=lr_vae,
                        lr_adv=lr_adv,
                        patience=_early_stop_patience,
                        class_weights=pos_weights,
                        model_type=_type_adv,
                        _dataset_name=_dataset_name,
                        optimizer=optimizer_vae,
                        verbose=False
                    )
                
                # Log métricas do VAE no Optuna
                if isinstance(vae_metrics, dict):
                    trial.set_user_attr("vae_recon_loss", vae_metrics.get('final_recon_loss', 0))
                    trial.set_user_attr("vae_kl_loss", vae_metrics.get('final_kl_loss', 0))
                
            except Exception as e:
                print(f"❌ Erro no treinamento do VAE (Trial {trial.number}): {e}")
                raise optuna.exceptions.TrialPruned()
            
            # Gerar dados desenviesados
            try:
                df_debiased_with_sensitive = generate_and_save_debiased_data_with_sensitive_info(
                    encoder_debiased, decoder_debiased, dataloader_data, scaler,
                    df_data_no_sensitive.columns, label_columns=target, 
                    sensitive_columns=sensitive_features, 
                    filename=f"{_path}_{_mitigation_tech}_{_dataset_name}_trial_{trial.number}.csv",
                    model_type=_type_adv
                )
                if fold_data.size > 0:
                    df_debiased_with_sensitive["fold"] = fold_data
                
                df_debiased_test_with_sensitive = generate_and_save_debiased_data_with_sensitive_info(
                    encoder_debiased, decoder_debiased, dataloader_test, scaler,
                    df_data_no_sensitive.columns, label_columns=target, 
                    sensitive_columns=sensitive_features,
                    filename=f"{_path}_{_mitigation_tech}_{_dataset_name}_test_trial_{trial.number}.csv",
                    model_type=_type_adv
                )
                if fold_test.size > 0:
                    df_debiased_test_with_sensitive["fold"] = fold_test
            except Exception as e:
                print(f"❌ Erro ao gerar dados desenviesados (Trial {trial.number}): {e}")
                raise optuna.exceptions.TrialPruned()
            
            # =================================================================
            # TREINAMENTO DA MLP
            # =================================================================
            y_debiased_stratify_keys = [dict_['target']] + dict_['sensitive_features']
            stratify_debiased_cv = df_debiased_with_sensitive[y_debiased_stratify_keys].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )
            X_debiased_cv = df_debiased_with_sensitive.drop(
                columns=[dict_['target']] + [c for c in non_feature_columns if c in df_debiased_with_sensitive.columns]
            )
            y_debiased_cv = df_debiased_with_sensitive[dict_['target']]
            
            stratify_debiased_test = df_debiased_test_with_sensitive[y_debiased_stratify_keys].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )
            X_debiased_test = df_debiased_test_with_sensitive.drop(
                columns=[dict_['target']] + [c for c in non_feature_columns if c in df_debiased_test_with_sensitive.columns]
            )
            y_debiased_test = df_debiased_test_with_sensitive[dict_['target']]
            fixed_validation_mask_debiased = (
                df_debiased_with_sensitive["fold"].astype(int) == int(_validation_fold)
                if "fold" in df_debiased_with_sensitive.columns
                else None
            )
            
            # Treinar MLP com hiperparâmetros otimizados
            try:
                balanced_acc = train_mlp(
                    _dataset_name=_dataset_name,
                    X_cv=X_debiased_cv,
                    y_cv=y_debiased_cv,
                    X_test_biased=X_debiased_test,
                    y_test=y_debiased_test,
                    stratify_cv=stratify_debiased_cv,
                    sensitive_features=sensitive_features,
                    _set_loss=_set_loss,
                    mitigation_tech=_mitigation_tech,
                    opt_type=optimizer_mlp,
                    batch_size=batch_size_mlp,
                    k_folds=_k_folds,
                    _epochs=epochs_mlp,
                    verbose=False,
                    fixed_validation_mask=fixed_validation_mask_debiased,
                )
                
                # Log informações do trial
                print(f"\n{'='*60}")
                print(f"📊 Trial {trial.number} - Balanced Accuracy: {balanced_acc:.4f}")
                print(f"🔧 VAE: lr={lr_vae:.6f}, lr_adv={lr_adv:.6f}, λ_adv={lambda_adv:.4f}, β={beta:.4f}")
                print(f"🔧 MLP: opt={optimizer_mlp}, batch={batch_size_mlp}, epochs={epochs_mlp}")
                print(f"{'='*60}\n")
                
            except Exception as e:
                print(f"❌ Erro no treinamento da MLP (Trial {trial.number}): {e}")
                raise optuna.exceptions.TrialPruned()
            
            return balanced_acc
        
    # =================================================================
    # EXECUTAR OTIMIZAÇÃO
    # =================================================================
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=_seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=3,
            interval_steps=1
        )
    )
    
    # Callback para salvar progresso
    def callback(study, trial):
        if trial.number % 10 == 0:
            df = study.trials_dataframe()
            df.to_csv(f"./results/optuna/optuna_progress_{_dataset_name}_{_mitigation_tech}.csv", index=False)
    
    study.optimize(objective, n_trials=_n_trials, callbacks=[callback], show_progress_bar=True)
    
    # =================================================================
    # RESULTADOS FINAIS
    # =================================================================
    print("\n" + "="*80)
    print("🏆 OTIMIZAÇÃO CONCLUÍDA")
    print("="*80)
    print(f"\n📊 Melhor Balanced Accuracy: {study.best_trial.value:.4f}")
    print(f"🔢 Número do melhor trial: {study.best_trial.number}")
    
    print(f"\n{'🔧 HIPERPARÂMETROS DO VAE':^80}")
    print("-"*80)
    for key, value in study.best_params.items():
        if 'vae' in key or key in ['lr_adv', 'lambda_adv', 'beta', 'batch_size_vae', 'optimizer_vae']:
            print(f"  • {key:.<30} {value}")
    
    print(f"\n{'🔧 HIPERPARÂMETROS DA MLP':^80}")
    print("-"*80)
    for key, value in study.best_params.items():
        if 'mlp' in key or key in ['epochs_mlp']:
            print(f"  • {key:.<30} {value}")
    
    # Análise de importância dos hiperparâmetros
    print(f"\n{'📊 IMPORTÂNCIA DOS HIPERPARÂMETROS':^80}")
    print("-"*80)
    try:
        importance = optuna.importance.get_param_importances(study)
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {param:.<30} {imp:.4f}")
    except:
        print("  Não foi possível calcular a importância dos hiperparâmetros")
    
    # Salvar resultados finais
    study.trials_dataframe().to_csv(
        f"./results/optuna/optuna_final_{_dataset_name}_{_mitigation_tech}.csv", 
        index=False
    )
    
    # Salvar melhores hiperparâmetros em JSON
    import json
    with open(f"./results/optuna/best_params_{_dataset_name}_{_mitigation_tech}.json", 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial': study.best_trial.number
        }, f, indent=4)
    
    print(f"\n💾 Resultados salvos em:")
    print(f"  • ./results/optuna/optuna_final_{_dataset_name}_{_mitigation_tech}.csv")
    print(f"  • ./results/optuna/best_params_{_dataset_name}_{_mitigation_tech}.json")
    
    return study.best_params, study.best_value

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Optuna Hyperparameter Optimization for VAE + MLP Pipeline")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["db-pad-ufes-20", "db-hiba", "db-midas"],
        help="Nome do dataset para processar"
    )


    args = parser.parse_args()
    _dataset_name = args.dataset

    best_params, best_value = optuna_run(_dataset_name)
    
    print(f"\n✅ Otimização concluída com sucesso!")
    print(f"📈 Melhor Balanced Accuracy: {best_value:.4f}")
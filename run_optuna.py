# -*- coding: utf-8 -*-
"""
Optuna hyperparameter search for the VAE + MLP pipeline.

Runs the PIP configuration (Pre + In + Post) end to end for every trial: DEMV
oversamples the training data, an adversarial VAE/AE strips the sensitive
information from the latent space, and an MLP classifies the reconstruction.
The balanced accuracy of the MLP is the value Optuna maximizes.

The best hyperparameters are written to
./results/optuna/best_params_{dataset}_PIP.json, the file main.py reads.

Usage:
    python run_optuna.py --dataset db-pad-ufes-20

Author: Matheus Becali Rocha
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

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
try:
    print(f"Device in use: {torch.cuda.get_device_name(device)}")
except Exception:
    print('No CUDA device found, falling back to CPU.')

# Global seed, shared by every stochastic step so runs are reproducible
_seed = 78645


def build_split_iterator(X_cv, stratify_cv, k_folds, fixed_validation_mask=None):
    """
    Return the list of train/validation splits used by the classifier.

    Mirrors the helper of main.py, so the search optimizes on exactly the same
    splits the final experiment uses:
    - fixed_validation_mask given: a single predefined split, where the samples
      flagged True become the validation set.
    - otherwise: a StratifiedKFold with k_folds splits, seeded by _seed.

    Args:
        X_cv: Features of the cross-validation set.
        stratify_cv: Composite key (target + sensitive attributes) used to stratify.
        k_folds: Number of folds when no fixed mask is given.
        fixed_validation_mask: Optional boolean mask flagging the validation samples.

    Returns:
        A tuple (splits, total_folds), where splits is a list of
        (train_indices, validation_indices) pairs.
    """
    if fixed_validation_mask is not None:
        mask = np.array(fixed_validation_mask, dtype=bool)
        train_idx = np.where(~mask)[0]
        valid_idx = np.where(mask)[0]

        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise ValueError(
                "Invalid fold split: the training or the validation set came out empty."
            )

        return [(train_idx, valid_idx)], 1

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=_seed)
    return list(kf.split(X_cv, stratify_cv)), k_folds

def generate_and_save_debiased_data_with_sensitive_info(encoder, decoder, dataloader, scaler, columns, 
                                                        label_columns, sensitive_columns, filename, model_type="vae", verbose=False):
    """
    Generate and save the debiased data together with the sensitive information.

    Every sample is pushed through the trained encoder/decoder pair, so the
    reconstruction carries as little sensitive information as the adversarial
    training managed to remove. The result is written back on the original scale
    and keeps the label and the sensitive columns.

    :param encoder: Trained encoder (VAE or AE).
    :param decoder: Trained decoder, reconstructing the features from the latent space.
    :param dataloader: Loader yielding (X_batch, s_batch, y_batch) tuples.
    :param scaler: Scaler fitted on the training set, used to invert the normalization.
    :param columns: Names of the feature columns of the reconstructed data.
    :param label_columns: Name of the label column.
    :param sensitive_columns: Names of the sensitive columns to append.
    :param filename: Path of the CSV file to write.
    :param model_type: 'vae' (encoder returns mean/log_var) or 'ae' (returns z).
    :param verbose: Enables the detailed log messages.
    :return: The DataFrame holding the debiased data.
    """
    encoder.eval()
    decoder.eval()
    debiased_outputs, labels_outputs, sensitive_outputs = [], [], []

    with torch.no_grad():
        for X_batch, s_batch, y_batch in dataloader:  # 's_batch' holds the sensitive data
            X_batch = X_batch.to(device)
            s_batch = s_batch.to(device)
            y_batch = y_batch.to(device)

            if model_type == "vae":
                # The mean is used instead of a sample from z: the reconstruction
                # has to be deterministic so the saved dataset is reproducible.
                mean, log_var = encoder(X_batch)
                # z = reparameterize(mean, log_var)
                recon_batch = decoder(mean)
                # recon_batch = decoder(mean)
            elif model_type == "ae":
                z = encoder(X_batch)
                recon_batch = decoder(z)
            else:
                raise ValueError("model_type must be 'vae' or 'ae'")
                
            debiased_outputs.append(recon_batch.cpu().numpy())
            labels_outputs.append(y_batch.cpu().numpy())
            sensitive_outputs.append(s_batch.cpu().numpy())

    X_debiased_scaled = np.concatenate(debiased_outputs, axis=0)

    X_debiased_original = X_debiased_scaled.copy()
    
    # Alternative: run inverse_transform ONLY on the non-binary columns
    # X_debiased_original[:, indices_nao_bin] = scaler.inverse_transform(
    #     X_debiased_scaled[:, indices_nao_bin]
    # )

    # Back to the original scale, so the saved CSV is readable and comparable
    X_debiased_original = scaler.inverse_transform(X_debiased_scaled)
    
    y_full = np.concatenate(labels_outputs, axis=0)
    # Concatenate the sensitive data
    s_full = np.concatenate(sensitive_outputs, axis=0)

    # Build the DataFrame with the debiased data plus the sensitive data
    df_debiased = pd.DataFrame(X_debiased_original, columns=columns)
    df_debiased[label_columns] = y_full

    # Append the sensitive columns, kept aside from the reconstruction
    for i, sensitive_column in enumerate(sensitive_columns):
        df_debiased[sensitive_column] = s_full[:, i]

    # Write the data to the CSV file
    df_debiased.to_csv(filename, index=False)

    if verbose:
        print(
            f"\nDebiased and sensitive data saved to '{filename}' (on the original scale).")

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
    Train an MLP with stratified cross-validation and return its balanced accuracy.

    This is the search-time twin of train_mlp in main.py: it exposes the
    hyperparameters Optuna tunes and, unlike its counterpart, writes no CSV,
    since only the returned score matters here.

    Args:
        _dataset_name: Dataset name (db-pad-ufes-20, db-hiba or db-midas).
        X_cv: Features of the cross-validation set, sensitive attributes included.
        y_cv: Labels of the cross-validation set.
        X_test_biased: Test features, sensitive attributes included.
        y_test: Test labels.
        stratify_cv: Composite key used to stratify the folds.
        sensitive_features: Names of the protected attributes.
        _set_loss: Loss function name, kept for signature compatibility.
        mitigation_tech: Mitigation technique (None, Pre, In, PI, PP, IP, Pos, PIP).
        opt_type: Optimizer, 'Adam', 'AdamW' or 'SGD'.
        batch_size: Batch size.
        k_folds: Number of folds when no fixed mask is given.
        _epochs: Maximum number of epochs.
        verbose: Whether to print the per-epoch progress.
        lr: Learning rate.
        hidden_size_ratio: input_size to hidden_size ratio (2 -> input_size // 2).
        weight_decay: L2 regularization factor.
        sched_factor: Learning-rate reduction factor of the scheduler.
        sched_patience: Scheduler patience, in epochs.
        fixed_validation_mask: Optional boolean mask flagging the validation samples.

    Returns:
        The mean balanced accuracy across the folds, or 0.0 when no fold produced
        a metric. This is the value Optuna maximizes.
    """

    # --- STRUCTURE HOLDING THE DETAILED PER-FOLD METRICS ---
    if _dataset_name in ["db-pad-ufes-20", "db-hiba", "db-midas"]:
        sp, di, eod, aod = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    else:
        raise NotImplementedError(f"Invalid Dataset: {_dataset_name}")

    # Initialize the split strategy (fixed fold or StratifiedKFold)
    split_iterator, total_folds = build_split_iterator(
        X_cv, stratify_cv, k_folds, fixed_validation_mask=fixed_validation_mask
    )

    # Lists holding the metrics of every fold
    metrics_list = []
    accuracy = []
    balancedAccuracyScore = []
    recall = []
    precision = []
    f1 = []
    auc = []
    test_loss = []

    # Validation loop: one iteration per fold
    for fold, (train_idx, valid_idx) in enumerate(split_iterator):
        # Early-stopping state: stop after `limit_stop` consecutive epochs
        # without an improvement of the validation loss
        curr_loss = 0
        limit_stop = 20
        prev_loss = np.inf
        trigger_times = 0

        if verbose:
            print(f"\nFold {fold+1}/{total_folds}")

        # Split the cross-validation set into training and validation for this fold
        X_train_biased = X_cv.iloc[train_idx].reset_index(drop=True)
        y_train = y_cv.iloc[train_idx]
        X_valid_biased = X_cv.iloc[valid_idx].reset_index(drop=True)
        y_valid = y_cv.iloc[valid_idx]

        # Drop the sensitive attributes from the predictors: the model must not
        # use them directly, but they are kept in X_*_biased for the fairness metrics
        X_train_no_sensitive = X_train_biased.drop(columns=sensitive_features)
        X_valid_no_sensitive = X_valid_biased.drop(columns=sensitive_features)
        X_test_no_sensitive = X_test_biased.drop(columns=sensitive_features)

        # Normalization: the scaler is fitted on the training fold only
        scaler = StandardScaler()
        X_train_scaler_np = scaler.fit_transform(X_train_no_sensitive)
        X_valid_scaler_np = scaler.transform(X_valid_no_sensitive)
        X_test_scaler_np = scaler.transform(X_test_no_sensitive)

        X_train_scaler = pd.DataFrame(X_train_scaler_np, columns=X_train_no_sensitive.columns, index=X_train_no_sensitive.index)
        X_valid_scaler = pd.DataFrame(X_valid_scaler_np, columns=X_valid_no_sensitive.columns, index=X_valid_no_sensitive.index)
        X_test_scaler = pd.DataFrame(X_test_scaler_np, columns=X_test_no_sensitive.columns, index=X_test_no_sensitive.index)

        # Convert to tensors before feeding the DataLoaders
        X_train_tensor = torch.tensor(np.array(X_train_scaler), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
        X_valid_tensor = torch.tensor(np.array(X_valid_scaler), dtype=torch.float32)
        y_valid_tensor = torch.tensor(y_valid.to_numpy(), dtype=torch.long)
        X_test_tensor = torch.tensor(np.array(X_test_scaler), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

        # Class weights compensate the class imbalance in the loss function
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

        # Network geometry derived from the data
        _input_size = X_train_scaler.shape[1]
        _hidden_size = _input_size // hidden_size_ratio  # tuned by Optuna
        _num_classes = len(np.unique(y_train))

        # Build the model (adding dropout would require changing ClassifyingNetwork)
        model = ClassifyingNetwork(
            input_size=_input_size,
            hidden_size=_hidden_size,
            num_classes=_num_classes
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer with a tunable learning rate and weight decay
        if opt_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"Invalid Optimizer: {opt_type}")

        # Scheduler with tunable parameters: drops the LR when the loss plateaus
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

            # Validation
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

        # Evaluation on the validation set (feeds the post-processing mitigator)
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

        # Evaluation on the test set
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
        
        # Preserve the original predictions before the sensitive-attribute loop,
        # so post-processing on one attribute cannot contaminate the next one
        y_pred_test_original = list(y_pred_test)
        y_proba_test_original = list(y_proba_test)

        # Compute the fairness metrics for EVERY attribute, whatever the strategy
        for sens_attr in sensitive_features:
            # Reset the predictions for each sensitive attribute
            y_pred_test = list(y_pred_test_original)

            A_test = X_test_biased[sens_attr]

            # Define both groups directly.
            # We assume the "privileged" group is 0 and the "unprivileged" one is 1.
            # Adapt this if your encoding is different.
            group_a_test = (A_test == 1)  # Unprivileged group (e.g. 'others', 'female')
            group_b_test = (A_test == 0)  # Privileged group (e.g. 'white', 'male')

            if mitigation_tech in ["Pos", "PP", "IP", "PIP"]:
                if verbose:
                    if mitigation_tech == "PIP":
                        print(f"Running with mitigation technique: {mitigation_tech} - Stage 3: Post-processing!")
                    elif mitigation_tech in ["PP", "IP"]:
                        print(f"Running with mitigation technique: {mitigation_tech} - Stage 2: Post-processing!")
                    else:
                        print(f"Running with mitigation technique: {mitigation_tech}!")
                    
                # Avoid data leakage: the mitigator is fitted on the validation set
                # and only then applied to the test set.
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

                # Use y_pred_test_cpp for the fairness metrics of this attribute
                y_pred_test = y_pred_test_cpp
            else:
                pass

            # Performance of this sensitive attribute; post-processing may have
            # changed the predictions, so it is recomputed per attribute.
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

            # The fairness metrics below are defined for binary problems only
            if np.array_equal(np.unique(y_true_test), [0, 1]):
                # Compute the disparity between the two groups
                sp_value = statistical_parity(group_a_test, group_b_test, y_pred_test)
                di_value = disparate_impact(group_a_test, group_b_test, y_pred_test)
                eod_value = equal_opportunity_diff(group_a_test, group_b_test, y_pred_test, y_true_test)
                aod_value = average_odds_diff(group_a_test, group_b_test, y_pred_test, y_true_test)


                # Absolute value: only the size of the disparity matters, not its sign
                sp_fold[sens_attr] = np.abs(sp_value)
                di_fold[sens_attr] = np.abs(di_value)
                eod_fold[sens_attr] = np.abs(eod_value)
                aod_fold[sens_attr] = np.abs(aod_value)

                # Also accumulate the disparity across folds, per attribute
                sp[sens_attr].append(np.abs(sp_value))
                di[sens_attr].append(np.abs(di_value))
                eod[sens_attr].append(np.abs(eod_value))
                aod[sens_attr].append(np.abs(aod_value))
            
            else:
                pass
            


        # Aggregate the fold metrics over every sensitive attribute.
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
    Run the Optuna hyperparameter search for the VAE + MLP pipeline.

    Loads the dataset, applies DEMV to the training split once, then runs
    _n_trials trials. Each trial samples a VAE/AE and MLP configuration, trains
    the whole pipeline and reports the balanced accuracy of the MLP.

    The results are saved as a progress CSV, a final CSV and a JSON file with
    the best hyperparameters.

    Args:
        _dataset_name: Dataset name (db-pad-ufes-20, db-hiba or db-midas).

    Returns:
        A tuple (best_params, best_value) with the best configuration found and
        its balanced accuracy.
    """
    _mitigation_tech = "PIP"
    _seed = 78645
    _n_trials = 200
    _num_epochs_vae = 200  # kept high enough for the VAE to converge per trial
    _early_stop_patience = 20
    _verbose = False
    _type_adv = "vae"


    # Sensitive-attribute types of each dataset, in the order the adversary expects
    if _dataset_name in ["db-pad-ufes-20"]:
        _attribute_types = ['binary', 'binary'] # gender, fitz
    elif _dataset_name in ["db-hiba", "db-midas"]:
        _attribute_types = ['binary'] # gender
    else:
        raise NotImplementedError(f"Invalid Dataset: {_dataset_name}")

    # Path where the debiased data of each trial is written
    _path = f"./debiased/{_dataset_name}/dados_desenviesados_com_sensitivos"
    _filename=f"{_path}_{_mitigation_tech}_{_dataset_name}.csv"
    os.makedirs(os.path.dirname(f"./debiased/{_dataset_name}"), exist_ok=True)
    os.makedirs(os.path.dirname(_path), exist_ok=True)

    # MLP arguments
    _k_folds = 5
    _epochs = 1
    _set_loss = "weighted_cross_entropy_loss"
    _validation_fold = 1  # Fixed validation fold, matching main.py
    ######################################################################################

    # Conversion helpers, identical to the ones in main.py
    def convert_fitzpatrick_scale(df):
        """Binarize the Fitzpatrick scale: [0,1,2] -> 0, [3,4,5] -> 1."""
        if 'fitzpatrick' in df.columns:
            df['fitzpatrick'] = df['fitzpatrick'].apply(lambda x: 0 if x < 3 else 1)
        return df

    def convert_diagnosis(df):
        """Binarize the diagnosis: "NC"/"benign" -> 0, anything else -> 1."""
        if 'diagnosis' in df.columns:
            df['diagnosis'] = df['diagnosis'].apply(lambda x: 0 if x == "NC" or x == "benign" else 1)
        return df

    # Read the datasets (train+val and test kept separate, as in main.py)
    df_data = pd.read_csv(f'./data/{_dataset_name}/processed_{_dataset_name}.csv', delimiter=',')
    df_data_test = pd.read_csv(f'./data/{_dataset_name}/processed_{_dataset_name}_test.csv', delimiter=',')

    if _dataset_name == "db-midas":
        # Clean and standardize the column names so they match features_setting()
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

    # Apply both conversions to both datasets
    df_data = convert_fitzpatrick_scale(df_data)
    df_data_test = convert_fitzpatrick_scale(df_data_test)
    df_data = convert_diagnosis(df_data)
    df_data_test = convert_diagnosis(df_data_test)

    # Simplified rule: the first two columns are always non-predictive (img_id, fold).
    non_feature_columns = list(df_data.columns[:2])

    if "fold" not in df_data.columns:
        raise ValueError("Column 'fold' was not found in df_data.")

    """Setup features: column groups of this dataset"""
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

    # Fixed split by fold, matching main.py
    train_mask = df_data["fold"].astype(int) != int(_validation_fold)
    val_mask = df_data["fold"].astype(int) == int(_validation_fold)

    df_train = df_data.loc[train_mask].copy()
    df_val = df_data.loc[val_mask].copy()

    # ==============================================================================
    # Running DEMV - ONLY on the TRAINING data, to avoid data leakage
    # ==============================================================================
    if _mitigation_tech in ["Pre", "PI", "PP", "PIP"]:
        if _verbose:
            print("Running DEMV (Pre-processing) on the training data only!")

        from demv import DEMV

        demv = DEMV(sensitive_vars=sensitive_features, round_level=1, verbose=False)
        demv_x = df_train.drop(
            columns=[target] + [c for c in non_feature_columns if c in df_train.columns]
        )
        demv_y = df_train[target]
        x_new, y_new = demv.fit_transform(demv_x, demv_y)
        # Rebuild df_train keeping the non-predictive columns.
        # DEMV oversamples, so `x_new` is larger than the original `df_train`.
        df_train_new = x_new.copy()
        df_train_new[target] = y_new.copy()

        # The "non_feature" columns (img_id, fold) must be restored so nothing
        # breaks downstream. For the original rows (the first N ones):
        n_orig = len(df_train)
        for c in non_feature_columns:
            if c in df_train.columns:
                col_data = list(df_train[c])
                
                # Fill the synthetic rows created by DEMV with default values
                if c == "fold":
                    # A dummy fold (-1) keeps the synthetic rows out of the validation set
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
        Objective function maximized by Optuna.

        Samples one hyperparameter configuration, trains the adversarial VAE/AE,
        generates the debiased data and trains the MLP on it.

        Args:
            trial: The Optuna trial supplying the sampled values.

        Returns:
            The balanced accuracy of the MLP for this configuration.

        Raises:
            optuna.exceptions.TrialPruned: when any stage of the pipeline fails,
            so a single bad configuration does not abort the whole study.
        """
        # =================================================================
        # VAE/AE HYPERPARAMETERS (search ranges differ per dataset)
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
        # MLP HYPERPARAMETERS (limited to the ones train_mlp accepts)
        # =================================================================
        batch_size_mlp = trial.suggest_categorical("batch_size_mlp", [16, 32, 64, 128])
        optimizer_mlp = trial.suggest_categorical("optimizer_mlp", ["Adam", "SGD"])
        # epochs_mlp = trial.suggest_int('epochs_mlp', 100, 500)  # tunable if needed
        epochs_mlp = 50  # fixed: keeps each trial cheap enough to run 200 of them
        
        # =================================================================
        # VAE/AE TRAINING (In-Processing)
        # =================================================================
        if _mitigation_tech in ["In", "PI", "IP", "PIP"]:
            # Prepare the data for PyTorch using the fold split, as in main.py
            X_train = np.array(df_train[feature_cols]).astype(np.float32)
            y_label_train = np.array(df_train[target]).astype(np.float32)
            y_sensitive_train = np.array(df_train[sensitive_features]).astype(np.float32)
            fold_train = df_train["fold"].astype(int).to_numpy() if "fold" in df_train.columns else np.array([])

            X_val = np.array(df_val[feature_cols]).astype(np.float32)
            y_label_val = np.array(df_val[target]).astype(np.float32)
            y_sensitive_val = np.array(df_val[sensitive_features]).astype(np.float32)
            fold_val = df_val["fold"].astype(int).to_numpy() if "fold" in df_val.columns else np.array([])

            # Separate test data
            df_data_test_no_sensitive = df_data_test[feature_cols]
            df_data_test_sensitive = df_data_test[sensitive_features]
            X_test = np.array(df_data_test_no_sensitive).astype(np.float32)
            y_label_test = np.array(df_data_test[target]).astype(np.float32)
            y_sensitive_test = np.array(df_data_test_sensitive).astype(np.float32)
            fold_test = df_data_test["fold"].astype(int).to_numpy() if "fold" in df_data_test.columns else np.array([])
            
            # Normalization: the scaler is fitted on the training split only
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
            
            # Combine train + val: the debiased dataset covers both splits
            X_data_normalized = np.concatenate([X_train, X_val], axis=0)
            y_sensitive_data = np.concatenate([y_sensitive_train, y_sensitive_val], axis=0)
            y_label_data = np.concatenate([y_label_train, y_label_val], axis=0)
            fold_data = np.concatenate([fold_train, fold_val], axis=0) if fold_train.size > 0 and fold_val.size > 0 else np.array([])
            dataloader_data = get_dataloader(X_data_normalized, y_sensitive_data, y_label_data, batch_size_vae, False)
            
            # VAE settings: the latent space is slightly smaller than the input
            latent_dims = int(X_train.shape[1] // 1.2)
            pos_weights = calculate_class_weights(y_sensitive_train, device, verbose=False)
            
            # Train the VAE/AE against the adversary
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
                
                # Record the VAE metrics on the Optuna trial
                if isinstance(vae_metrics, dict):
                    trial.set_user_attr("vae_recon_loss", vae_metrics.get('final_recon_loss', 0))
                    trial.set_user_attr("vae_kl_loss", vae_metrics.get('final_kl_loss', 0))
                
            except Exception as e:
                print(f"[ERROR] VAE training failed (Trial {trial.number}): {e}")
                raise optuna.exceptions.TrialPruned()
            
            # Generate the debiased data the MLP will be trained on
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
                print(f"[ERROR] Debiased data generation failed (Trial {trial.number}): {e}")
                raise optuna.exceptions.TrialPruned()
            
            # =================================================================
            # MLP TRAINING
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
            
            # Train the MLP with the sampled hyperparameters
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
                
                # Log the trial information
                print(f"\n{'='*60}")
                print(f"Trial {trial.number} - Balanced Accuracy: {balanced_acc:.4f}")
                print(f"VAE: lr={lr_vae:.6f}, lr_adv={lr_adv:.6f}, lambda_adv={lambda_adv:.4f}, beta={beta:.4f}")
                print(f"MLP: opt={optimizer_mlp}, batch={batch_size_mlp}, epochs={epochs_mlp}")
                print(f"{'='*60}\n")
                
            except Exception as e:
                print(f"[ERROR] MLP training failed (Trial {trial.number}): {e}")
                raise optuna.exceptions.TrialPruned()
            
            return balanced_acc
        
    # =================================================================
    # RUN THE OPTIMIZATION
    # =================================================================
    # TPESampler seeded for reproducibility; MedianPruner drops trials that
    # already look worse than the median of the finished ones
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=_seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=3,
            interval_steps=1
        )
    )
    
    # Callback saving the progress every 10 trials, so a crash loses little
    def callback(study, trial):
        if trial.number % 10 == 0:
            df = study.trials_dataframe()
            df.to_csv(f"./results/optuna/optuna_progress_{_dataset_name}_{_mitigation_tech}.csv", index=False)
    
    study.optimize(objective, n_trials=_n_trials, callbacks=[callback], show_progress_bar=True)
    
    # =================================================================
    # FINAL RESULTS
    # =================================================================
    print("\n" + "="*80)
    print("OPTIMIZATION FINISHED")
    print("="*80)
    print(f"\nBest Balanced Accuracy: {study.best_trial.value:.4f}")
    print(f"Best trial number: {study.best_trial.number}")
    
    print(f"\n{'VAE HYPERPARAMETERS':^80}")
    print("-"*80)
    for key, value in study.best_params.items():
        if 'vae' in key or key in ['lr_adv', 'lambda_adv', 'beta', 'batch_size_vae', 'optimizer_vae']:
            print(f"  - {key:.<30} {value}")
    
    print(f"\n{'MLP HYPERPARAMETERS':^80}")
    print("-"*80)
    for key, value in study.best_params.items():
        if 'mlp' in key or key in ['epochs_mlp']:
            print(f"  - {key:.<30} {value}")
    
    # Hyperparameter importance analysis
    print(f"\n{'HYPERPARAMETER IMPORTANCE':^80}")
    print("-"*80)
    try:
        importance = optuna.importance.get_param_importances(study)
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {param:.<30} {imp:.4f}")
    except:
        print("  Could not compute the hyperparameter importance")
    
    # Save the final results
    study.trials_dataframe().to_csv(
        f"./results/optuna/optuna_final_{_dataset_name}_{_mitigation_tech}.csv", 
        index=False
    )
    
    # Save the best hyperparameters as JSON: this is the file main.py reads
    import json
    with open(f"./results/optuna/best_params_{_dataset_name}_{_mitigation_tech}.json", 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial': study.best_trial.number
        }, f, indent=4)
    
    print("\nResults saved to:")
    print(f"  - ./results/optuna/optuna_final_{_dataset_name}_{_mitigation_tech}.csv")
    print(f"  - ./results/optuna/best_params_{_dataset_name}_{_mitigation_tech}.json")
    
    return study.best_params, study.best_value

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Optuna Hyperparameter Optimization for VAE + MLP Pipeline")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["db-pad-ufes-20", "db-hiba", "db-midas"],
        help="Name of the dataset to process"
    )


    args = parser.parse_args()
    _dataset_name = args.dataset

    best_params, best_value = optuna_run(_dataset_name)
    
    print("\nOptimization finished successfully!")
    print(f"Best Balanced Accuracy: {best_value:.4f}")
"""
Bias mitigation pipeline for skin-lesion classification.

Runs the full experiment for one dataset, one mitigation technique and one
classifier: it applies the pre-processing (DEMV), in-processing (adversarial
VAE) and post-processing (MLDebiaser) stages requested, trains the chosen
classifier and writes the per-fold performance and fairness metrics to CSV.

Mitigation techniques, where P = Pre, I = In and the trailing P = Post:
    None, Pre, In, Pos, PI, PP, IP, PIP

Usage:
    python main.py --dataset db-pad-ufes-20 --mitigation None --classify mlp
    python main.py --dataset db-hiba --mitigation PIP --classify mlp --validation_fold 3

Author: Matheus Becali Rocha
Email: matheusbecali@gmail.com
"""

# ================================================================================================ #
# Libraries
# ================================================================================================ #
import argparse
import os

import numpy as np
import pandas as pd
import scipy.stats as ss
import torch
import torch.nn.functional as F
from demv import DEMV

# for binaryclass
from holisticai.bias.metrics import (
    average_odds_diff,
    disparate_impact,
    equal_opportunity_diff,
    statistical_parity,
)
from holisticai.bias.mitigation import MLDebiaser
from sklearn import tree
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.net import ClassifyingNetwork, train_debiased_autoencoder, train_debiased_vae
from utils.helpers import (
    calculate_class_weights,
    evaluate_fairness_latent,
    features_setting,
    load_best_hyperparameters,
    prepare_data_loader,
    save_results_to_csv,
)

# ================================================================================================ #

# GPU Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
try:
    print(f"Device in use: {torch.cuda.get_device_name(device)}")
except Exception:
    print('No CUDA device found, falling back to CPU.')

# Global seed, shared by every stochastic step so runs are reproducible
_seed = 78645

# ================================================================================================ #

def build_split_iterator(X_cv, stratify_cv, k_folds, fixed_validation_mask=None):
    """
    Return the list of train/validation splits used by the classifiers.

    Two strategies are supported:
    - fixed_validation_mask given: a single predefined split, where the samples
      flagged True become the validation set. Used when the fold comes from the
      dataset's own 'fold' column, keeping every experiment on the same split.
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

# ================================================================================================ #

def generate_and_save_debiased_data_with_sensitive_info(encoder, decoder, dataloader, scaler, columns, 
                                                        label_columns, sensitive_columns, filename, 
                                                        model_type="vae", verbose=False):
    """
    Generate and save the debiased data together with the sensitive information.

    Every sample is pushed through the trained encoder/decoder pair, so the
    reconstruction carries as little sensitive information as the adversarial
    training managed to remove. The result is written back on the original scale
    and keeps the label and the sensitive columns, so the downstream classifier
    can still compute the fairness metrics.

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
                # The mean is used instead of a sample from z,, the reconstruction
                # has to be deterministic so the saved dataset is reproducible.
                mean, _ = encoder(X_batch)
                recon_batch = decoder(mean)
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

# ================================================================================================ #

def train_mlp(_dataset_name, X_cv, y_cv, X_test_biased, y_test, stratify_cv, sensitive_features, 
              _set_loss, mitigation_tech, opt_type="Adam", batch_size=32, k_folds=5, 
              _epochs=2000, verbose=True,
              lr=0.001, weight_decay=0.001,
              fixed_validation_mask=None,
              validation_fold_value=None):
    """
    Train and evaluate an MLP classifier, fold by fold.

    For every fold the routine normalizes the data, trains the network with a
    class-weighted cross-entropy and early stopping, then evaluates it on the
    test set. When the mitigation technique includes a post-processing stage
    (Pos, PP, IP, PIP), the MLDebiaser is fitted on the validation set and
    applied to the test predictions, once per sensitive attribute.

    Args:
        _dataset_name: Dataset name, used to build the output path.
        X_cv: Features of the cross-validation set, sensitive attributes included.
        y_cv: Labels of the cross-validation set.
        X_test_biased: Test features, sensitive attributes included.
        y_test: Test labels.
        stratify_cv: Composite key used to stratify the folds.
        sensitive_features: Names of the protected attributes.
        _set_loss: Loss function name, recorded in the CSV.
        mitigation_tech: Mitigation technique (None, Pre, In, PI, PP, IP, Pos, PIP).
        opt_type: Optimizer, 'Adam', 'AdamW' or 'SGD'.
        batch_size: Batch size.
        k_folds: Number of folds when no fixed mask is given.
        _epochs: Maximum number of epochs.
        verbose: Whether to print the per-epoch progress.
        lr: Learning rate.
        weight_decay: L2 regularization factor.
        fixed_validation_mask: Optional boolean mask flagging the validation samples.
        validation_fold_value: Fold number recorded in the CSV.

    Returns:
        The mean balanced accuracy across the folds, or 0.0 when no fold produced
        a metric.
    """

    if _dataset_name not in ["db-pad-ufes-20", "db-hiba", "db-midas"]:
        raise NotImplementedError(f"Invalid Dataset: {_dataset_name}")

    # Initialize the split strategy (fixed fold or StratifiedKFold)
    split_iterator, total_folds = build_split_iterator(
        X_cv, stratify_cv, k_folds, fixed_validation_mask=fixed_validation_mask
    )

    # Lists holding the metrics of every fold
    accuracy = []
    balancedAccuracyScore = []
    recall = []
    precision = []
    f1 = []
    test_loss = []

    # Validation loop: one iteration per fold
    for fold, (train_idx, valid_idx) in enumerate(split_iterator):
        # Early-stopping state: stop after `limit_stop` consecutive epochs
        # without an improvement of the validation loss
        curr_loss = 0
        limit_stop = 20 #100
        prev_loss = np.inf
        trigger_times = 0

        if verbose:
            print(f"\nFold {fold+1}/{total_folds}")

        # Split the cross-validation set into training and validation for this fold
        X_train_biased = X_cv.iloc[train_idx].reset_index(drop=True)
        y_train = y_cv.iloc[train_idx]  # positionally aligned with X_cv
        X_valid_biased = X_cv.iloc[valid_idx].reset_index(drop=True)
        y_valid = y_cv.iloc[valid_idx]


        # Drop the sensitive attributes from the predictors: the model must not
        # use them directly, but they are kept in X_*_biased for the fairness metrics
        X_train_no_sensitive = X_train_biased.drop(columns=sensitive_features)
        X_valid_no_sensitive = X_valid_biased.drop(columns=sensitive_features)
        X_test_no_sensitive = X_test_biased.drop(columns=sensitive_features)
        # columns = X_train_no_sensitive.columns

        ######
        # 1: Normalization
        ######
        scaler = StandardScaler()
        X_train_scaler_np = scaler.fit_transform(X_train_no_sensitive) 
        X_valid_scaler_np = scaler.transform(X_valid_no_sensitive)
        X_test_scaler_np  = scaler.transform(X_test_no_sensitive)

        # Back to DataFrames, preserving the column names and the index
        X_train_scaler = pd.DataFrame(X_train_scaler_np, columns=X_train_no_sensitive.columns, index=X_train_no_sensitive.index)
        X_valid_scaler = pd.DataFrame(X_valid_scaler_np, columns=X_valid_no_sensitive.columns, index=X_valid_no_sensitive.index)
        X_test_scaler  = pd.DataFrame(X_test_scaler_np, columns=X_test_no_sensitive.columns, index=X_test_no_sensitive.index)

        # Convert to tensors before feeding the DataLoaders
        X_train_tensor = torch.tensor(np.array(X_train_scaler), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
        X_valid_tensor = torch.tensor(np.array(X_valid_scaler), dtype=torch.float32)
        y_valid_tensor = torch.tensor(y_valid.to_numpy(), dtype=torch.long)
        X_test_tensor  = torch.tensor(np.array(X_test_scaler), dtype=torch.float32)
        y_test_tensor  = torch.tensor(y_test.to_numpy(), dtype=torch.long)

        # Class weights compensate the class imbalance in the loss function
        y_train_np = y_train_tensor.cpu().numpy()  # Back to NumPy before computing the weights
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_np), y=y_train_np)
        # class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        # Build the DataLoaders for training, validation and test
        train_dataloader, _ = prepare_data_loader(
            X_train_tensor,
            y_train_tensor,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        valid_dataloader, _ = prepare_data_loader(X_valid_tensor, y_valid_tensor, batch_size=batch_size, shuffle=False)
        test_dataloader, _ = prepare_data_loader(X_test_tensor, y_test_tensor, batch_size=batch_size, shuffle=False)

        # Network geometry derived from the data: the hidden layer is half the input
        _input_size = X_train_scaler.shape[1]
        _hidden_size = X_train_scaler.shape[1] // 2
        _num_classes = len(np.unique(y_train))

        # Criar e treinar modelo
        model = ClassifyingNetwork(input_size=_input_size, 
                                    hidden_size=_hidden_size, 
                                    num_classes=_num_classes)

        model = model.to(device)

        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        if opt_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise NotImplementedError(f"Invalid Optimizer: {opt_type}")
        
        _sched_factor = 0.1 
        _sched_min_lr = 1e-6
        _sched_patience = 20 #10

        scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            factor=_sched_factor, 
                                                            min_lr=_sched_min_lr,
                                                            patience=_sched_patience)

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
                    # print("Current LR:", scheduler_lr.get_last_lr())
                    print(f'Epoch [{epoch+1}/{_epochs}]')
                    print(f"Valid -> Loss: {valid_epoch_loss:.4f}, BACC: {valid_bacc:.4f}")
                    print(f"Train -> Loss: {train_epoch_loss:.4f}, BACC: {train_bacc:.4f}")

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

        # Evaluation on the test set
        model.eval()
        test_total_loss = 0.0
        y_true_valid_pp, y_proba_valid = [], []
        with torch.no_grad():
            for X_batch, y_batch in valid_dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                probs = F.softmax(outputs, dim=1)
                y_true_valid_pp.extend(y_batch.tolist())
                y_proba_valid.extend(probs.cpu().numpy())

        with torch.no_grad():
            y_true_test, y_pred_test, y_proba_test = [], [], []
            for X_batch, y_batch in test_dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)

                probs = F.softmax(outputs, dim=1)  # logits turned into probabilities
                _, predicted = torch.max(outputs.data, 1)

                te_loss = criterion(outputs, y_batch)
                test_total_loss += te_loss.item()

                y_true_test.extend(y_batch.tolist())
                y_pred_test.extend(predicted.cpu().tolist())
                y_proba_test.extend(probs.cpu().numpy())  # keeps the (n, 2) matrix

        test_epoch_loss = test_total_loss/len(test_dataloader)

        if _dataset_name in ["db-pad-ufes-20", "db-hiba", "db-midas"]:
            sp_fold, di_fold, eod_fold, aod_fold = {}, {}, {}, {}

        # Performance metrics, one entry per sensitive attribute
        accuracy_fold, bacc_fold = {}, {}
        precision_fold, recall_fold, f1_fold = {}, {}, {}
        
        # Preserve the original predictions before the sensitive-attribute loop,
        # so post-processing on one attribute cannot contaminate the next one.
        y_pred_test_original = list(y_pred_test)
        y_proba_test_original = list(y_proba_test)

        # Compute the fairness metrics for EVERY attribute, whatever the strategy
        for sens_attr in sensitive_features:
            # Reset the predictions for each sensitive attribute
            y_pred_test = list(y_pred_test_original)

            A_test = X_test_biased[sens_attr]

            # Define both groups directly.
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
                    
                # The mitigator is fitted on the validation set and only then applied to the test set.
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

                y_pred_test = y_pred_test_cpp
            else:
                pass
            
            # Performance of this sensitive attribute
            # Post-processing may have changed the predictions, so it is recomputed per attribute
            accuracy_fold[sens_attr] = accuracy_score(y_true_test, y_pred_test)
            bacc_fold[sens_attr] = balanced_accuracy_score(y_true_test, y_pred_test)
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
            else:
                pass

        # Aggregate the fold metrics into the function return value.
        if len(bacc_fold) > 0:
            test_loss.append(test_epoch_loss)
            accuracy.append(float(np.mean(list(accuracy_fold.values()))))
            balancedAccuracyScore.append(float(np.mean(list(bacc_fold.values()))))
            precision.append(float(np.mean(list(precision_fold.values()))))
            recall.append(float(np.mean(list(recall_fold.values()))))
            f1.append(float(np.mean(list(f1_fold.values()))))
    
        # Save the per-fold results (performance averaged over the sensitive attributes)
        accuracy_fold_mean = float(np.mean([accuracy_fold[s] for s in sensitive_features if s in accuracy_fold])) if len(accuracy_fold) > 0 else np.nan
        bacc_fold_mean = float(np.mean([bacc_fold[s] for s in sensitive_features if s in bacc_fold])) if len(bacc_fold) > 0 else np.nan
        precision_fold_mean = float(np.mean([precision_fold[s] for s in sensitive_features if s in precision_fold])) if len(precision_fold) > 0 else np.nan
        recall_fold_mean = float(np.mean([recall_fold[s] for s in sensitive_features if s in recall_fold])) if len(recall_fold) > 0 else np.nan
        f1_fold_mean = float(np.mean([f1_fold[s] for s in sensitive_features if s in f1_fold])) if len(f1_fold) > 0 else np.nan

        fold_csv_filename = f'./results/classification_model/mlp/{_dataset_name}_per_fold.csv'
        os.makedirs(os.path.dirname(fold_csv_filename), exist_ok=True)
        fold_results_data = {
            'Fold': int(validation_fold_value) if validation_fold_value is not None else (fold + 1),
            'Dataset': _dataset_name,
            'Mitigation Technic': mitigation_tech,
            '_batch_size': batch_size,
            'Loss Function': _set_loss,
            'Test - loss': test_epoch_loss,
            'Test - Accuracy Score': accuracy_fold_mean,
            'Test - Balanced Accuracy Score': bacc_fold_mean,
            'Test - Precision Score': precision_fold_mean,
            'Test - Recall Score': recall_fold_mean,
            'Test - F1 Score': f1_fold_mean,
        }

        # Add the fairness columns dynamically, one per (metric, attribute) pair
        if _dataset_name in ["db-pad-ufes-20", "db-hiba", "db-midas"]:
            all_fold_metrics = {'Statistical Parity': sp_fold, 
                                    'Disparate Impact': di_fold, 
                                    'Equal Opportunity Diff': eod_fold, 
                                    'Average Odds Diff': aod_fold,
                                    }

        # Outer loop iterates over the SENSITIVE ATTRIBUTES
        for sens_attr in sensitive_features:
            # Inner loop iterates over the METRIC TYPES
            for metric_name, fold_metric_dict in all_fold_metrics.items():
                # Value of the current attribute/metric pair
                value = fold_metric_dict.get(sens_attr, 'N/A')
                
                # Store it in the results dict under the formatted column name
                fold_results_data[f'{metric_name} ({sens_attr})'] = value

        fold_header = list(fold_results_data.keys())

        save_results_to_csv(fold_csv_filename, fold_results_data, fold_header)

    print(f"Per-fold results saved to {fold_csv_filename}")

    return float(np.mean(balancedAccuracyScore)) if len(balancedAccuracyScore) > 0 else 0.0

# ================================================================================================ #

class GiniDistance:
    """
    Computes the Gini-based distance for a dataset.
    - Uses ranks of the data to calculate Gini distances.
    - Supports a customizable Gini parameter (`gini_param`).

    Working on ranks rather than raw values makes the distance robust to
    outliers and to features on wildly different scales, which is why it is
    used as the KNN metric here.
    """
    def __init__(self, X, gini_param=2):
        self.X = X
        self.gini_param = gini_param

    def _rank(self, X):
        """
        Compute ranks for the given data array.
        Used to calculate the cumulative ranks for Gini distance.
        """
        ranks = np.apply_along_axis(ss.rankdata, 0, X)
        return X.shape[0] - ranks + 1

    def compute_gini_ranks(self, X):
        """
        Compute cumulative ranks for training and test data.
        Adjusts ranks based on the Gini parameter.
        """
        X_cat = np.concatenate((self.X, X), axis=0)
        ranks = (self._rank(X_cat) / X_cat.shape[0] * self.X.shape[0]) ** (self.gini_param - 1)
        return ranks[:self.X.shape[0]], ranks[self.X.shape[0]:]

    def gini_distance(self, x, Y, decum_rank_x, decum_ranks_Y):
        """
        Calculate the Gini distance between a single point `x` and a set of points `Y`.
        Combines rank differences with feature differences to compute the distance.
        """
        distance = -np.sum((x - Y) * (decum_rank_x - decum_ranks_Y), axis=1)
        return distance

    def compute_distances(self, X):
        """
        Compute the Gini distance matrix for test data `X` relative to training data.
        Returns a precomputed distance matrix.
        """
        ranks_train, ranks_test = self.compute_gini_ranks(X)
        distances = np.zeros((X.shape[0], self.X.shape[0]))
        
        for i, x in enumerate(X):
            distances[i, :] = self.gini_distance(x, self.X, ranks_test[i], ranks_train)
        return distances

def train_knn(_dataset_name, X_cv, y_cv, X_test_biased, y_test, stratify_cv, sensitive_features, 
               mitigation_tech, k_folds=5, curvature=0.1, n_neighbors=3, verbose=True,
               fixed_validation_mask=None,
               validation_fold_value=None):
    """
    Train and evaluate a KNN classifier using the Gini distance, fold by fold.

    Instead of the Euclidean metric, the neighbours are ranked by the
    rank-based Gini distance computed by GiniDistance, passed to scikit-learn as
    a precomputed distance matrix. The post-processing stage and the metric
    bookkeeping follow the same protocol as train_mlp.

    Args:
        _dataset_name: Dataset name, used to build the output path.
        X_cv: Features of the cross-validation set, sensitive attributes included.
        y_cv: Labels of the cross-validation set.
        X_test_biased: Test features, sensitive attributes included.
        y_test: Test labels.
        stratify_cv: Composite key used to stratify the folds.
        sensitive_features: Names of the protected attributes.
        mitigation_tech: Mitigation technique (None, Pre, In, PI, PP, IP, Pos, PIP).
        k_folds: Number of folds when no fixed mask is given.
        curvature: Kept for signature compatibility; unused by this implementation.
        n_neighbors: Number of neighbours of the KNN.
        verbose: Whether to print the per-fold progress.
        fixed_validation_mask: Optional boolean mask flagging the validation samples.
        validation_fold_value: Fold number recorded in the CSV.

    Returns:
        The mean balanced accuracy across the folds, or 0.0 when no fold produced
        a metric.
    """

    if _dataset_name not in ["db-pad-ufes-20", "db-hiba", "db-midas"]:
        raise NotImplementedError(f"Invalid Dataset: {_dataset_name}")
    
    # Initialize the split strategy (fixed fold or StratifiedKFold)
    split_iterator, total_folds = build_split_iterator(
        X_cv, stratify_cv, k_folds, fixed_validation_mask=fixed_validation_mask
    )

    # Lists holding the metrics of every fold
    accuracy = []
    balancedAccuracyScore = []
    recall = []
    precision = []
    f1 = []

    # Validation loop: one iteration per fold
    for fold, (train_idx, valid_idx) in enumerate(split_iterator):

        if verbose:
            print(f"\nFold {fold+1}/{total_folds}")

        # Split the cross-validation set into training and validation for this fold
        X_train_biased = X_cv.iloc[train_idx].reset_index(drop=True)
        y_train = y_cv.iloc[train_idx]  # positionally aligned with X_cv
        X_valid_biased = X_cv.iloc[valid_idx].reset_index(drop=True)
        y_valid = y_cv.iloc[valid_idx]


        # Drop the sensitive attributes from the predictors: the model must not
        # use them directly, but they are kept in X_*_biased for the fairness metrics
        X_train_no_sensitive = X_train_biased.drop(columns=sensitive_features)
        X_valid_no_sensitive = X_valid_biased.drop(columns=sensitive_features)
        X_test_no_sensitive = X_test_biased.drop(columns=sensitive_features)
        # columns = X_train_no_sensitive.columns

        ######
        # 1: Normalization
        ######
        scaler = StandardScaler()
        X_train_scaler_np = scaler.fit_transform(X_train_no_sensitive) 
        X_valid_scaler_np = scaler.transform(X_valid_no_sensitive)
        X_test_scaler_np  = scaler.transform(X_test_no_sensitive)

        # Back to DataFrames, preserving the column names and the index
        X_train_scaler = pd.DataFrame(X_train_scaler_np, columns=X_train_no_sensitive.columns, index=X_train_no_sensitive.index)
        X_valid_scaler = pd.DataFrame(X_valid_scaler_np, columns=X_valid_no_sensitive.columns, index=X_valid_no_sensitive.index)
        X_test_scaler  = pd.DataFrame(X_test_scaler_np, columns=X_test_no_sensitive.columns, index=X_test_no_sensitive.index)

        # Back to NumPy arrays, the format expected by scikit-learn
        X_train_np = np.array(X_train_scaler)
        y_train_np = y_train.to_numpy()
        X_valid_np = np.array(X_valid_scaler)
        y_valid_np = y_valid.to_numpy()
        X_test_np  = np.array(X_test_scaler)
        y_test_np  = y_test.to_numpy()

        #  Compute Gini distance
        gini_calculator = GiniDistance(X_train_np, gini_param=2)
        train_distances_gini = gini_calculator.compute_distances(
            X_train_np)
        valid_distances_gini = gini_calculator.compute_distances(
            X_valid_np)
        test_distances_gini = gini_calculator.compute_distances(
            X_test_np)

        # Constant columns yield undefined ranks; the sentinels keep those pairs
        # far apart instead of letting a NaN propagate into the KNN
        train_distances_gini = np.nan_to_num(
            train_distances_gini, nan=100, posinf=100, neginf=-100)
        valid_distances_gini = np.nan_to_num(
            valid_distances_gini, nan=100, posinf=100, neginf=-100)
        test_distances_gini = np.nan_to_num(
            test_distances_gini, nan=100, posinf=100, neginf=-100)

        # print(np.unique(train_distances_gini))

        # metric='precomputed': scikit-learn receives the Gini distance matrix
        # directly instead of computing a distance of its own
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
        knn.fit(train_distances_gini, y_train)

        y_pred_train = knn.predict(train_distances_gini)
        y_pred_valid = knn.predict(valid_distances_gini)
        y_pred_test = knn.predict(test_distances_gini)
        y_proba_valid = knn.predict_proba(valid_distances_gini)
        y_proba_test = knn.predict_proba(test_distances_gini)
        
        if verbose:
            print(f"Train -> BACC: {balanced_accuracy_score(y_train_np, y_pred_train):.4f}")
            print(f"Valid -> BACC: {balanced_accuracy_score(y_valid_np, y_pred_valid):.4f}")
            print(f"Test -> BACC: {balanced_accuracy_score(y_test_np, y_pred_test):.4f}")

        if _dataset_name in ["db-pad-ufes-20", "db-hiba", "db-midas"]:
            sp_fold, di_fold, eod_fold, aod_fold = {}, {}, {}, {}

        # Performance metrics, one entry per sensitive attribute
        accuracy_fold, balancedAccuracyScore_fold = {}, {}
        precision_fold, recall_fold, f1_fold = {}, {}, {}
        
        # Preserve the original predictions before the sensitive-attribute loop,
        # so post-processing on one attribute cannot contaminate the next one
        y_pred_test_original = np.array(y_pred_test).copy()
        y_proba_test_original = np.array(y_proba_test).copy()

        # Compute the fairness metrics
        for sens_attr in sensitive_features:
            # Reset the predictions for each sensitive attribute
            y_pred_test = y_pred_test_original.copy()

            A_test = X_test_biased[sens_attr]

            # Define both groups directly
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
                    
                # The mitigator is fitted on the validation set and only then applied to the test set
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

                y_pred_test = y_pred_test_cpp
            else:
                pass

            # Performance of this sensitive attribute
            # Post-processing may have changed the predictions, so it is recomputed per attribute
            accuracy_fold[sens_attr] = accuracy_score(y_test_np, y_pred_test)
            balancedAccuracyScore_fold[sens_attr] = balanced_accuracy_score(y_test_np, y_pred_test)
            recall_fold[sens_attr] = recall_score(y_test_np, y_pred_test, average='weighted')
            precision_fold[sens_attr] = precision_score(
                y_test_np,
                y_pred_test,
                average='weighted',
                zero_division=True,
            )
            f1_fold[sens_attr] = f1_score(y_test_np, y_pred_test, average='weighted')
            
            # The fairness metrics below are defined for binary problems only
            if np.array_equal(np.unique(y_test_np), [0, 1]):
                # Compute the disparity between the two groups
                sp_value = statistical_parity(group_a_test, group_b_test, y_pred_test)
                di_value = disparate_impact(group_a_test, group_b_test, y_pred_test)
                eod_value = equal_opportunity_diff(group_a_test, group_b_test, y_pred_test, y_test_np)
                aod_value = average_odds_diff(group_a_test, group_b_test, y_pred_test, y_test_np)

                # Absolute value: only the size of the disparity matters, not its sign
                sp_fold[sens_attr] = np.abs(sp_value)
                di_fold[sens_attr] = np.abs(di_value)
                eod_fold[sens_attr] = np.abs(eod_value)
                aod_fold[sens_attr] = np.abs(aod_value)


        # Aggregate the fold metrics into the function return value.
        if len(balancedAccuracyScore_fold) > 0:
            accuracy.append(float(np.mean(list(accuracy_fold.values()))))
            balancedAccuracyScore.append(float(np.mean(list(balancedAccuracyScore_fold.values()))))
            precision.append(float(np.mean(list(precision_fold.values()))))
            recall.append(float(np.mean(list(recall_fold.values()))))
            f1.append(float(np.mean(list(f1_fold.values()))))

        # Save the per-fold results (performance averaged over the sensitive attributes)
        accuracy_fold_mean = float(np.mean([accuracy_fold[s] for s in sensitive_features if s in accuracy_fold])) if len(accuracy_fold) > 0 else np.nan
        bacc_fold_mean = float(np.mean([balancedAccuracyScore_fold[s] for s in sensitive_features if s in balancedAccuracyScore_fold])) if len(balancedAccuracyScore_fold) > 0 else np.nan
        precision_fold_mean = float(np.mean([precision_fold[s] for s in sensitive_features if s in precision_fold])) if len(precision_fold) > 0 else np.nan
        recall_fold_mean = float(np.mean([recall_fold[s] for s in sensitive_features if s in recall_fold])) if len(recall_fold) > 0 else np.nan
        f1_fold_mean = float(np.mean([f1_fold[s] for s in sensitive_features if s in f1_fold])) if len(f1_fold) > 0 else np.nan

        fold_csv_filename = f'./results/classification_model/knn/{_dataset_name}_per_fold.csv'
        os.makedirs(os.path.dirname(fold_csv_filename), exist_ok=True)
        fold_results_data = {
            'Fold': int(validation_fold_value) if validation_fold_value is not None else (fold + 1),
            'Dataset': _dataset_name,
            'Mitigation Technic': mitigation_tech,
            'Test - Accuracy Score': accuracy_fold_mean,
            'Test - Balanced Accuracy Score': bacc_fold_mean,
            'Test - Precision Score': precision_fold_mean,
            'Test - Recall Score': recall_fold_mean,
            'Test - F1 Score': f1_fold_mean,
        }

        # Add the fairness columns dynamically, one per (metric, attribute) pair
        if _dataset_name in ["db-pad-ufes-20", "db-hiba", "db-midas"]:
            all_fold_metrics = {'Statistical Parity': sp_fold, 
                                'Disparate Impact': di_fold, 
                                'Equal Opportunity Diff': eod_fold, 
                                'Average Odds Diff': aod_fold,
                                }
            
        # Outer loop iterates over the SENSITIVE ATTRIBUTES
        for sens_attr in sensitive_features:
            # Inner loop iterates over the METRIC TYPES
            for metric_name, fold_metric_dict in all_fold_metrics.items():
                # Value of the current attribute/metric pair
                value = fold_metric_dict.get(sens_attr, 'N/A')
                # Store it in the results dict under the formatted column name
                fold_results_data[f'{metric_name} ({sens_attr})'] = value

        fold_header = list(fold_results_data.keys())

        save_results_to_csv(fold_csv_filename, fold_results_data, fold_header)
  
    print(f"Per-fold results saved to {fold_csv_filename}")

    return float(np.mean(balancedAccuracyScore)) if len(balancedAccuracyScore) > 0 else 0.0

# ================================================================================================ #

def train_dtree(_dataset_name, X_cv, y_cv, X_test_biased, y_test, stratify_cv, sensitive_features, 
                mitigation_tech, k_folds=5, verbose=True, max_depth=3,
                fixed_validation_mask=None,
                validation_fold_value=None):
    """
    Train and evaluate a Decision Tree classifier, fold by fold.

    The tree splits on the Gini criterion and is depth-limited to keep it
    interpretable. The post-processing stage and the metric bookkeeping follow
    the same protocol as train_mlp.

    Args:
        _dataset_name: Dataset name, used to build the output path.
        X_cv: Features of the cross-validation set, sensitive attributes included.
        y_cv: Labels of the cross-validation set.
        X_test_biased: Test features, sensitive attributes included.
        y_test: Test labels.
        stratify_cv: Composite key used to stratify the folds.
        sensitive_features: Names of the protected attributes.
        mitigation_tech: Mitigation technique (None, Pre, In, PI, PP, IP, Pos, PIP).
        k_folds: Number of folds when no fixed mask is given.
        verbose: Whether to print the per-fold progress.
        max_depth: Maximum depth of the tree.
        fixed_validation_mask: Optional boolean mask flagging the validation samples.
        validation_fold_value: Fold number recorded in the CSV.

    Returns:
        The mean balanced accuracy across the folds, or 0.0 when no fold produced
        a metric.
    """

    if _dataset_name not in ["db-pad-ufes-20", "db-hiba", "db-midas"]:
        raise NotImplementedError(f"Invalid Dataset: {_dataset_name}")
    
    # Initialize the split strategy (fixed fold or StratifiedKFold)
    split_iterator, total_folds = build_split_iterator(
        X_cv, stratify_cv, k_folds, fixed_validation_mask=fixed_validation_mask
    )

    # Lists holding the metrics of every fold
    accuracy = []
    balancedAccuracyScore = []
    recall = []
    precision = []
    f1 = []

    # Validation loop: one iteration per fold
    for fold, (train_idx, valid_idx) in enumerate(split_iterator):

        if verbose:
            print(f"\nFold {fold+1}/{total_folds}")

        # Split the cross-validation set into training and validation for this fold
        X_train_biased = X_cv.iloc[train_idx].reset_index(drop=True)
        y_train = y_cv.iloc[train_idx]  # positionally aligned with X_cv
        X_valid_biased = X_cv.iloc[valid_idx].reset_index(drop=True)
        y_valid = y_cv.iloc[valid_idx]


        # Drop the sensitive attributes from the predictors: the model must not
        # use them directly, but they are kept in X_*_biased for the fairness metrics
        X_train_no_sensitive = X_train_biased.drop(columns=sensitive_features)
        X_valid_no_sensitive = X_valid_biased.drop(columns=sensitive_features)
        X_test_no_sensitive = X_test_biased.drop(columns=sensitive_features)
        # columns = X_train_no_sensitive.columns

        ######
        # 1: Normalization
        ######
        scaler = StandardScaler()
        X_train_scaler_np = scaler.fit_transform(X_train_no_sensitive) 
        X_valid_scaler_np = scaler.transform(X_valid_no_sensitive)
        X_test_scaler_np  = scaler.transform(X_test_no_sensitive)

        # Back to DataFrames, preserving the column names and the index
        X_train_scaler = pd.DataFrame(X_train_scaler_np, columns=X_train_no_sensitive.columns, index=X_train_no_sensitive.index)
        X_valid_scaler = pd.DataFrame(X_valid_scaler_np, columns=X_valid_no_sensitive.columns, index=X_valid_no_sensitive.index)
        X_test_scaler  = pd.DataFrame(X_test_scaler_np, columns=X_test_no_sensitive.columns, index=X_test_no_sensitive.index)

        # Back to NumPy arrays, the format expected by scikit-learn
        X_train_np = np.array(X_train_scaler)
        y_train_np = y_train.to_numpy()
        X_valid_np = np.array(X_valid_scaler)
        y_valid_np = y_valid.to_numpy()
        X_test_np  = np.array(X_test_scaler)
        y_test_np  = y_test.to_numpy()

        # Decision Tree
        arvore_gini = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = max_depth)
        arvore_gini.fit(X_train_np, y_train_np)

        # training set
        y_pred_train = arvore_gini.predict(X_train_np)

        # validation set
        y_pred_valid = arvore_gini.predict(X_valid_np)
        y_proba_valid = arvore_gini.predict_proba(X_valid_np)

        # test set
        y_pred_test = arvore_gini.predict(X_test_np)
        y_proba_test = arvore_gini.predict_proba(X_test_np)
        
        if verbose:
            print(f"Train -> BACC: {balanced_accuracy_score(y_train_np, y_pred_train):.4f}")
            print(f"Valid -> BACC: {balanced_accuracy_score(y_valid_np, y_pred_valid):.4f}")
            print(f"Test -> BACC: {balanced_accuracy_score(y_test_np, y_pred_test):.4f}")

        if _dataset_name in ["db-pad-ufes-20", "db-hiba", "db-midas"]:
            sp_fold, di_fold, eod_fold, aod_fold = {}, {}, {}, {}

        # Performance metrics, one entry per sensitive attribute
        accuracy_fold, balancedAccuracyScore_fold = {}, {}
        precision_fold, recall_fold, f1_fold = {}, {}, {}
        
        # Preserve the original predictions before the sensitive-attribute loop,
        # so post-processing on one attribute cannot contaminate the next one
        y_pred_test_original = np.array(y_pred_test).copy()
        y_proba_test_original = np.array(y_proba_test).copy()

        # Compute the fairness metrics for EVERY attribute, whatever the strategy
        for sens_attr in sensitive_features:
            # Reset the predictions for each sensitive attribute
            y_pred_test = y_pred_test_original.copy()

            A_test = X_test_biased[sens_attr]

            # Define both groups directly.
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
                
                # The mitigator is fitted on the validation set and only then applied to the test set.
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
                
                y_pred_test = y_pred_test_cpp
            else:
                pass

            # Performance of this sensitive attribute
            # Post-processing may have changed the predictions, so it is recomputed per attribute.
            accuracy_fold[sens_attr] = accuracy_score(y_test_np, y_pred_test)
            balancedAccuracyScore_fold[sens_attr] = balanced_accuracy_score(y_test_np, y_pred_test)
            recall_fold[sens_attr] = recall_score(y_test_np, y_pred_test, average='weighted')
            precision_fold[sens_attr] = precision_score(
                y_test_np,
                y_pred_test,
                average='weighted',
                zero_division=True,
            )
            f1_fold[sens_attr] = f1_score(y_test_np, y_pred_test, average='weighted')
            
            # The fairness metrics below are defined for binary problems only
            if np.array_equal(np.unique(y_test_np), [0, 1]):
                # Compute the disparity between the two groups
                sp_value = statistical_parity(group_a_test, group_b_test, y_pred_test)
                di_value = disparate_impact(group_a_test, group_b_test, y_pred_test)
                eod_value = equal_opportunity_diff(group_a_test, group_b_test, y_pred_test, y_test_np)
                aod_value = average_odds_diff(group_a_test, group_b_test, y_pred_test, y_test_np)

                # Absolute value: only the size of the disparity matters, not its sign
                sp_fold[sens_attr] = np.abs(sp_value)
                di_fold[sens_attr] = np.abs(di_value)
                eod_fold[sens_attr] = np.abs(eod_value)
                aod_fold[sens_attr] = np.abs(aod_value)

            else:
                pass
            


        # Aggregate the fold metrics into the function return value.
        if len(balancedAccuracyScore_fold) > 0:
            accuracy.append(float(np.mean(list(accuracy_fold.values()))))
            balancedAccuracyScore.append(float(np.mean(list(balancedAccuracyScore_fold.values()))))
            recall.append(float(np.mean(list(recall_fold.values()))))
            precision.append(float(np.mean(list(precision_fold.values()))))
            f1.append(float(np.mean(list(f1_fold.values()))))

        # Save the per-fold results (performance averaged over the sensitive attributes)
        accuracy_fold_mean = float(np.mean([accuracy_fold[s] for s in sensitive_features if s in accuracy_fold])) if len(accuracy_fold) > 0 else np.nan
        bacc_fold_mean = float(np.mean([balancedAccuracyScore_fold[s] for s in sensitive_features if s in balancedAccuracyScore_fold])) if len(balancedAccuracyScore_fold) > 0 else np.nan
        precision_fold_mean = float(np.mean([precision_fold[s] for s in sensitive_features if s in precision_fold])) if len(precision_fold) > 0 else np.nan
        recall_fold_mean = float(np.mean([recall_fold[s] for s in sensitive_features if s in recall_fold])) if len(recall_fold) > 0 else np.nan
        f1_fold_mean = float(np.mean([f1_fold[s] for s in sensitive_features if s in f1_fold])) if len(f1_fold) > 0 else np.nan

        fold_csv_filename = f'./results/classification_model/dtree/{_dataset_name}_per_fold.csv'
        os.makedirs(os.path.dirname(fold_csv_filename), exist_ok=True)
        fold_results_data = {
            'Fold': int(validation_fold_value) if validation_fold_value is not None else (fold + 1),
            'Dataset': _dataset_name,
            'Mitigation Technic': mitigation_tech,
            'Test - Accuracy Score': accuracy_fold_mean,
            'Test - Balanced Accuracy Score': bacc_fold_mean,
            'Test - Precision Score': precision_fold_mean,
            'Test - Recall Score': recall_fold_mean,
            'Test - F1 Score': f1_fold_mean,
        }

        # Add the fairness columns dynamically, one per (metric, attribute) pair
        if _dataset_name in ["db-pad-ufes-20", "db-hiba", "db-midas"]:
            all_fold_metrics = {'Statistical Parity': sp_fold, 
                                'Disparate Impact': di_fold, 
                                'Equal Opportunity Diff': eod_fold, 
                                'Average Odds Diff': aod_fold}

        # Outer loop iterates over the SENSITIVE ATTRIBUTES
        for sens_attr in sensitive_features:
            # Inner loop iterates over the METRIC TYPES
            for metric_name, fold_metric_dict in all_fold_metrics.items():
                # Value of the current attribute/metric pair
                value = fold_metric_dict.get(sens_attr, 'N/A')
                
                # Store it in the results dict under the formatted column name
                fold_results_data[f'{metric_name} ({sens_attr})'] = value

        fold_header = list(fold_results_data.keys())

        save_results_to_csv(fold_csv_filename, fold_results_data, fold_header)

    print(f"Per-fold results saved to {fold_csv_filename}")

    return float(np.mean(balancedAccuracyScore)) if len(balancedAccuracyScore) > 0 else 0.0

# ================================================================================================ #

def main(_dataset_name = "db-pad-ufes-20", _num_epochs_vae=20000, _early_stop_patience=50, 
         _mitigation_tech="None", classify_type="mlp", _type_adv="vae", _verbose=True,
         _validation_fold=1):
    """
    Main entry point of the bias mitigation pipeline.

    The technique code spells out which stages run, where P = Pre-processing
    (DEMV), I = In-processing (adversarial VAE/AE) and the trailing P =
    Post-processing (MLDebiaser):
        None -> no mitigation (baseline)      PI  -> Pre + In
        Pre  -> DEMV only                     PP  -> Pre + Post
        In   -> adversarial VAE/AE only       IP  -> In + Post
        Pos  -> MLDebiaser only               PIP -> Pre + In + Post

    Args:
        _dataset_name: Dataset name (db-pad-ufes-20, db-hiba or db-midas).
        _num_epochs_vae: Maximum number of epochs of the adversarial VAE/AE.
        _early_stop_patience: Epochs without improvement before early stopping.
        _mitigation_tech: Mitigation technique (None, Pre, In, PI, PP, IP, Pos, PIP).
        classify_type: Classifier type (mlp, knn or dtree).
        _type_adv: Debiasing model, 'vae' or 'ae'.
        _verbose: Whether to print the detailed information.
        _validation_fold: Fold of the 'fold' column held out for validation.
    """
    
    print("="*80)
    print(f"Starting the pipeline for dataset: {_dataset_name}")
    print(f"Mitigation technique: {_mitigation_tech}")
    print("="*80)
    
    # The hyperparameters of the PIP run are reused by every technique, so the
    # comparison is not confounded by a different search per configuration.
    json_path = f"./results/optuna/best_params_{_dataset_name}_PIP.json"
    if classify_type == "mlp":
        lr_vae, lr_adv, lambda_adv, beta, batch_size_vae, optimizer_vae, batch_size_mlp, optimizer_mlp, _ = load_best_hyperparameters(json_path)
    else:
        lr_vae, lr_adv, lambda_adv, beta, batch_size_vae, optimizer_vae, _, _, _ = load_best_hyperparameters(json_path)

    # Sensitive-attribute types of each dataset, in the order the adversary expects
    if _dataset_name in ["db-pad-ufes-20"]:
        _attribute_types = ['binary', 'binary'] # gender, fitz
    elif _dataset_name in ["db-hiba", "db-midas"]:
        _attribute_types = ['binary'] # gender
    else:
        raise NotImplementedError(f"Invalid Dataset: {_dataset_name}")
    
    # Path where the debiased data is written
    _path = f"./debiased/{_dataset_name}/debiased_data_with_sensitive"
    _filename=f"{_path}_{_mitigation_tech}_{_dataset_name}.csv"
    os.makedirs(os.path.dirname(_filename), exist_ok=True)

    # MLP arguments
    _k_folds = 5
    _set_loss = "weighted_cross_entropy_loss"
    epochs_mlp = 2 # 2000
    ######################################################################################

    # Binarizes the Fitzpatrick scale from [0,1,2,3,4,5] down to [0,1]
    def convert_fitzpatrick_scale(df):
        """
        Convert the Fitzpatrick scale from 6 classes to 2 classes (binarization).
        [0, 1, 2] -> 0 (lighter skin, the privileged group)
        [3, 4, 5] -> 1 (darker skin, the unprivileged group)
        """
        if 'fitzpatrick' in df.columns:
            df['fitzpatrick'] = df['fitzpatrick'].apply(lambda x: 0 if x < 3 else 1)
        return df

    def convert_diagnosis(df):
        """
        Convert the diagnosis from string to 2 classes (binarization).
        "NC" / "benign" -> 0 (benign lesion)
        anything else   -> 1 (malignant lesion)
        """
        if 'diagnosis' in df.columns:
            df['diagnosis'] = df['diagnosis'].apply(lambda x: 0 if x == "NC" or x == "benign" else 1)
        return df
    
    ######################################################################################

    # Read the dataset csv file into a DataFrame
    df_data = pd.read_csv(f'./data/{_dataset_name}/processed_{_dataset_name}.csv', delimiter=',')
    df_data_test = pd.read_csv(f'./data/{_dataset_name}/processed_{_dataset_name}_test.csv', delimiter=',')

    if _dataset_name == "db-midas":
        # Clean and standardize the column names so they match features_setting()
        df_data.columns = (
            df_data.columns
            .str.strip()                            # strip leading/trailing whitespace
            .str.replace(r"\s+", "_", regex=True)   # inner whitespace becomes "_"
            .str.lower()                            # everything lowercase
        )
        df_data_test.columns = (
            df_data_test.columns
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.lower()
        )
    
    # Apply the Fitzpatrick conversion to both datasets
    df_data = convert_fitzpatrick_scale(df_data)
    df_data_test = convert_fitzpatrick_scale(df_data_test)

    # Apply the diagnosis conversion to both datasets
    df_data = convert_diagnosis(df_data)
    df_data_test = convert_diagnosis(df_data_test)

    # Simplified rule: the first two columns are always non-predictive (img_id, fold).
    non_feature_columns = list(df_data.columns[:2])

    if "fold" not in df_data.columns:
        raise ValueError("Column 'fold' was not found in df_data.")

    if _validation_fold not in set(df_data["fold"].astype(int).unique().tolist()):
        raise ValueError(
            f"validation_fold={_validation_fold} does not exist in the dataset. Available folds: "
            f"{sorted(df_data['fold'].astype(int).unique().tolist())}"
        )

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


    # In-Processing
    if _mitigation_tech in ["In", "PI", "IP", "PIP"]:
        if _verbose:
            if _mitigation_tech in ["PI", "PIP"]:
                print(f"Running with mitigation technique: {_mitigation_tech} - Stage 2: VAE (In-processing)!")
            elif _mitigation_tech == "IP":
                print(f"Running with mitigation technique: {_mitigation_tech} - Stage 1: VAE (In-processing)!")
            else:
                print(f"Running with mitigation technique: {_mitigation_tech}!")
        
        # Fixed split by fold: training = every fold but one, validation = the chosen fold
        train_mask = df_data["fold"].astype(int) != int(_validation_fold)
        val_mask = df_data["fold"].astype(int) == int(_validation_fold)

        df_train = df_data.loc[train_mask].copy()
        df_val = df_data.loc[val_mask].copy()

        # Running DEMV - ONLY on the TRAINING data
        if _mitigation_tech in ["Pre", "PI", "PP", "PIP"]:
            if _verbose:
                if _mitigation_tech in ["PI", "PP", "PIP"]:
                    print(f"Running with mitigation technique: {_mitigation_tech} - Stage 1: DEMV (Pre-processing)!")
                else:
                    print(f"Running with mitigation technique: {_mitigation_tech}!")

            demv = DEMV(sensitive_vars=sensitive_features, round_level=1, verbose=False)
            demv_x = df_train.drop(
                columns=[target] + [c for c in non_feature_columns if c in df_train.columns]
            )
            demv_y = df_train[target]
            x_new, y_new = demv.fit_transform(demv_x, demv_y)
            print('Maximum number of iterations: ', demv.get_iters())

            # Rebuild df_train keeping the non-predictive columns.
            # DEMV oversamples, so `x_new` is larger than the original `df_train`.
            df_train_new = x_new.copy()
            df_train_new[target] = y_new.copy()

            # The "non_feature" columns (img_id, fold) must be restored so nothing
            # breaks downstream. For the original rows (the first N ones):
            n_new = len(df_train_new)
            for c in non_feature_columns:
                if c in df_train.columns:
                    col_data = list(df_train[c])[:n_new]

                    # Pad to exactly the number of rows of df_train_new.
                    if len(col_data) < n_new:
                        pad_size = n_new - len(col_data)
                        if c == "fold":
                            # A dummy fold (-1) keeps the synthetic rows out of the validation set
                            col_data += [-1] * pad_size
                        else:
                            col_data += ["synthetic"] * pad_size

                    df_train_new[c] = col_data
            
            df_train = df_train_new.copy()
        
        feature_cols = [
            c for c in normal_features
            if c not in non_feature_columns and c in df_data.columns
        ]

        X_train = np.array(df_train[feature_cols]).astype(np.float32)
        y_label_train = np.array(df_train[target]).astype(np.float32)
        y_sensitive_train = np.array(df_train[sensitive_features]).astype(np.float32)
        fold_train = df_train["fold"].astype(int).to_numpy()

        X_val = np.array(df_val[feature_cols]).astype(np.float32)
        y_label_val = np.array(df_val[target]).astype(np.float32)
        y_sensitive_val = np.array(df_val[sensitive_features]).astype(np.float32)
        fold_val = df_val["fold"].astype(int).to_numpy()

        # --- BUILDING THE COMPOSITE STRATIFICATION KEY ---
        # Target and sensitive attributes joined into one string, so the splits
        # preserve the joint distribution of class and protected group.
        y_stratify_keys = [dict_['target']] + dict_['sensitive_features']
        y_stratify = df_data[y_stratify_keys].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        y_stratify_train = df_train[y_stratify_keys].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        y_stratify_val = df_val[y_stratify_keys].apply(lambda x: '_'.join(x.astype(str)), axis=1)

        # Prepare the test data from the already separated df_data_test
        df_data_test_no_sensitive = df_data_test[feature_cols]
        df_data_test_sensitive = df_data_test[sensitive_features]
        df_data_test[target] = df_data_test[target].astype(float)

        X_test = np.array(df_data_test_no_sensitive).astype(np.float32)
        y_label_test = np.array(df_data_test[target]).astype(np.float32)
        y_sensitive_test = np.array(df_data_test_sensitive).astype(np.float32)
        fold_test = df_data_test["fold"].astype(int).to_numpy() if "fold" in df_data_test.columns else np.array([])

        y_stratify_test = df_data_test[y_stratify_keys].apply(lambda x: '_'.join(x.astype(str)), axis=1)

        # Checking the stratification
        if _verbose:
            print("Checking the distribution of the stratified data...\n")

        # Percentage distribution of each dataset split
        original_dist = y_stratify.value_counts(normalize=True).sort_index() * 100
        train_dist = y_stratify_train.value_counts(normalize=True).sort_index() * 100
        val_dist = y_stratify_val.value_counts(normalize=True).sort_index() * 100
        test_dist = y_stratify_test.value_counts(normalize=True).sort_index() * 100

        # Combine the series into a single DataFrame for an easier comparison
        comparison_df = pd.DataFrame({
            "Original %": original_dist,
            "Train %": train_dist,
            "Validation %": val_dist,
            "Test %": test_dist
        })

        # Fill with 0 the rare case of a group missing from one of the splits
        comparison_df.fillna(0, inplace=True)

        # Print the formatted comparison table
        if _verbose:
            print("Group percentage distribution comparison table:")
            print(comparison_df.round(2))
            print(f"Fixed validation fold: {_validation_fold}")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        def get_dataloader(X, sensitive_attrs, labels, batch_size=32, _shuffle=True, _drop_last=False):
            dataset = TensorDataset(torch.from_numpy(X), 
                                    torch.from_numpy(sensitive_attrs), 
                                    torch.from_numpy(labels))
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=_shuffle,
                drop_last=_drop_last,
            )

        dataloader_train = get_dataloader(
            X_train,
            y_sensitive_train,
            y_label_train,
            batch_size=batch_size_vae,
            _shuffle=True,
            _drop_last=True, # avoids a final batch of 1 sample
        )
        dataloader_val = get_dataloader(X_val, y_sensitive_val, y_label_val, batch_size=batch_size_vae, _shuffle=False)
        dataloader_test = get_dataloader(X_test, y_sensitive_test, y_label_test, batch_size=batch_size_vae, _shuffle=False)

        X_data_normalized = np.concatenate([X_train, X_val], axis=0)
        y_sensitive_data = np.concatenate([y_sensitive_train, y_sensitive_val], axis=0)
        y_label_data = np.concatenate([y_label_train, y_label_val], axis=0)
        fold_data = np.concatenate([fold_train, fold_val], axis=0)

        dataloader_data = get_dataloader(X_data_normalized, y_sensitive_data, y_label_data, _shuffle=False)

        if _verbose:
            print("\n", 10 * "-", "Data shapes", 10 * "-")
            print(f"Training data shape: {X_train.shape}")
            print(f"Validation data shape: {X_val.shape}")
            print(f"Test data shape: {X_test.shape}")

        ######################################################################################

        # VAE arguments
        # The latent space is half the input size
        
        # _latent_dims = int(X_train.shape[1] // 1.2)
        _latent_dims = int(X_train.shape[1] // 2)
        # print(f"_latent_dims: {_latent_dims}")
        _pos_weights = calculate_class_weights(y_sensitive_train, device, verbose=_verbose)

        # Run In-Processing: trains the encoder/decoder against an adversary that
        # tries to recover the sensitive attributes from the latent space
        if _type_adv == "vae":
            encoder_debiased, decoder_debiased, _, _ = train_debiased_vae(
                train_loader=dataloader_train,
                val_loader=dataloader_val,
                input_dim=X_train.shape[1],
                attribute_types=_attribute_types,
                latent_dim=_latent_dims,
                num_epochs=_num_epochs_vae,
                patience=_early_stop_patience,
                lambda_adv=lambda_adv,
                beta_vae=beta,
                lr_adv=lr_adv,
                lr=lr_vae,
                model_type=_type_adv,
                class_weights=_pos_weights,
                optimizer=optimizer_vae,
                _dataset_name=_dataset_name,
                mitigation_type=_mitigation_tech,
                verbose=True
            )
        elif _type_adv == "ae":
            encoder_debiased, decoder_debiased, _, _ = train_debiased_autoencoder(
                train_loader=dataloader_train,
                val_loader=dataloader_val,
                input_dim=X_train.shape[1],
                attribute_types=_attribute_types,
                latent_dim=_latent_dims,
                num_epochs=_num_epochs_vae,
                lambda_adv=lambda_adv,
                lr=lr_vae,
                lr_adv=lr_adv,
                patience=_early_stop_patience,
                class_weights=_pos_weights,
                optimizer=optimizer_vae,
                model_type=_type_adv,
                _dataset_name=_dataset_name,
                mitigation_type=_mitigation_tech,
                verbose=True
            )
        else:
            raise ValueError("model_type must be 'vae' or 'ae'")

        ######################################################################################

        results = evaluate_fairness_latent(
            encoder=encoder_debiased,
            X=X_test,
            y_sensitive=y_sensitive_test,
            device=device,
            dataset_name=_dataset_name,
            mitigation_type=_mitigation_tech,
            title="Latent Space",
            model_type=_type_adv
        )

        ######################################################################################

        df_debiased_with_sensitive = generate_and_save_debiased_data_with_sensitive_info(
            encoder_debiased, decoder_debiased, dataloader_data, scaler,
            feature_cols, label_columns = target, sensitive_columns=sensitive_features,
            filename=_filename, model_type=_type_adv
        )
        df_debiased_with_sensitive["fold"] = fold_data

        _filename_test = f"{_path}_{_mitigation_tech}_{_dataset_name}_test.csv"

        df_debiased_test_with_sensitive = generate_and_save_debiased_data_with_sensitive_info(
            encoder_debiased, decoder_debiased, dataloader_test, scaler,
            feature_cols, label_columns = target, sensitive_columns=sensitive_features,
            filename=_filename_test, model_type=_type_adv
        )
        if fold_test.size > 0:
            df_debiased_test_with_sensitive["fold"] = fold_test

        if _verbose:
            aux_outputs = []
            aux_labels = []
            for X_aux, s_aux, y_aux in dataloader_data:  # 's_aux' holds the sensitive data
                aux_outputs.append(X_aux)
                aux_labels.append(y_aux)

            aux_x = np.concatenate(aux_outputs, axis=0)
            aux_y = np.concatenate(aux_labels, axis=0)
            df_aux = pd.DataFrame(aux_x, columns=feature_cols)
            df_aux[target] = aux_y

            
            if _dataset_name in ["db-pad-ufes-20", "db-hiba"]:
                print(pd.concat([
                    df_aux['age'].describe().rename('orig'),
                    df_debiased_with_sensitive['age'].describe().rename('filtered')
                ], axis=1))
            elif _dataset_name == "fairndb":
                pass
        
        # MLP
        df_debiased_no_sensitive = df_debiased_with_sensitive[normal_features]
        df_debiased_sensitive = df_debiased_with_sensitive[sensitive_features]
        df_debiased_with_sensitive[target] = df_debiased_with_sensitive[target].astype(float)

        df_debiased_test_no_sensitive = df_debiased_test_with_sensitive[normal_features]
        df_debiased_test_sensitive = df_debiased_test_with_sensitive[sensitive_features]
        df_debiased_test_with_sensitive[target] = df_debiased_test_with_sensitive[target].astype(float)

        # Building the stratification key and the initial split
        
        # --- BUILDING THE COMPOSITE STRATIFICATION KEY ---
        # Target and sensitive attributes joined into one string for stratification
        y_debiased_stratify_keys = [dict_['target']] + dict_['sensitive_features']

        # The sensitive variables are passed along: the classifier drops them
        # from the predictors itself, but still needs them for the fairness metrics
        stratify_debiased_cv = df_debiased_with_sensitive[y_debiased_stratify_keys].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        X_debiased_cv = df_debiased_with_sensitive.drop(
            columns=[dict_['target']] + [c for c in non_feature_columns if c in df_debiased_with_sensitive.columns]
        )
        y_debiased_cv = df_debiased_with_sensitive[dict_['target']]

        # print(df_debiased_test_no_sensitive)
        stratify_debiased_test = df_debiased_test_with_sensitive[y_debiased_stratify_keys].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        X_debiased_test = df_debiased_test_with_sensitive.drop(
            columns=[dict_['target']] + [c for c in non_feature_columns if c in df_debiased_test_with_sensitive.columns]
        )
        y_debiased_test = df_debiased_test_with_sensitive[dict_['target']]
        fixed_validation_mask_debiased = (
            df_debiased_with_sensitive["fold"].astype(int) == int(_validation_fold)
            if "fold" in df_debiased_with_sensitive.columns
            else None
        )

        if _verbose:
            print(f"Cross-Validation set size: {X_debiased_cv.shape}")
            print(f"Final test set size: {X_debiased_test.shape}\n")

            # Percentage distribution of each dataset split
            cv_dist_debiased = stratify_debiased_cv.value_counts(normalize=True).sort_index()
            test_dist_debiased = stratify_debiased_test.value_counts(normalize=True).sort_index()

            # Combine the distributions into a single DataFrame for comparison
            comparison_df_debiased = pd.DataFrame({
                "Cross-Validation (%)": cv_dist_debiased * 100,
                "Test (%)": test_dist_debiased * 100
            })

            # Comparison table
            print("Group percentage distribution comparison table:")
            print(comparison_df_debiased.round(2))

            print("="*80)
            print("Running MLP for classification")
            print("="*80)

        if classify_type == "mlp":
            train_mlp(_dataset_name, X_debiased_cv, y_debiased_cv, 
                    X_debiased_test, y_debiased_test, stratify_debiased_cv, 
                    sensitive_features, _set_loss, mitigation_tech=_mitigation_tech,
                    opt_type=optimizer_mlp, batch_size=batch_size_mlp, k_folds=_k_folds, 
                    _epochs=epochs_mlp,
                    fixed_validation_mask=fixed_validation_mask_debiased,
                    validation_fold_value=_validation_fold)
        elif classify_type == "knn":
            train_knn(_dataset_name, X_debiased_cv, y_debiased_cv, 
                    X_debiased_test, y_debiased_test, stratify_debiased_cv, 
                    sensitive_features, mitigation_tech=_mitigation_tech,
                    fixed_validation_mask=fixed_validation_mask_debiased,
                    validation_fold_value=_validation_fold)
        elif classify_type == "dtree":
            train_dtree(_dataset_name, X_debiased_cv, y_debiased_cv, 
                    X_debiased_test, y_debiased_test, stratify_debiased_cv, 
                    sensitive_features, mitigation_tech=_mitigation_tech, max_depth=3,
                    fixed_validation_mask=fixed_validation_mask_debiased,
                    validation_fold_value=_validation_fold)

    elif _mitigation_tech in ["Pre", "Pos", "PP", "None"]:
        
        
        y_stratify_keys = [dict_['target']] + dict_['sensitive_features']

        # ==============================================================================
        # Running DEMV - ONLY on the TRAINING data, to avoid data leakage
        # ==============================================================================
        if _mitigation_tech in ["Pre", "PP"]:
            if _verbose:
                if _mitigation_tech == "PP":
                    print(f"Running with mitigation technique: {_mitigation_tech} - Stage 1: DEMV (Pre-processing)!")
                else:
                    print(f"Running with mitigation technique: {_mitigation_tech}!")

            # Separate training and validation by fold before running DEMV
            train_mask_pre = df_data["fold"].astype(int) != int(_validation_fold)
            val_mask_pre = df_data["fold"].astype(int) == int(_validation_fold)

            df_train_pre = df_data.loc[train_mask_pre].copy()
            df_val_pre = df_data.loc[val_mask_pre].copy()

            demv = DEMV(sensitive_vars=sensitive_features, round_level=1, verbose=False)
            demv_x = df_train_pre.drop(
                columns=[target] + [c for c in non_feature_columns if c in df_train_pre.columns]
            )
            demv_y = df_train_pre[target]
            x_new, y_new = demv.fit_transform(demv_x, demv_y)
            print('Maximum number of iterations: ', demv.get_iters())

            # Rebuild df_train with the DEMV output, keeping the non-predictive columns
            df_train_new = x_new.copy()
            df_train_new[target] = y_new.copy()

            # Restore the "non_feature" columns (img_id, fold) so nothing breaks
            n_new = len(df_train_new)
            for c in non_feature_columns:
                if c in df_train_pre.columns:
                    col_data = list(df_train_pre[c])[:n_new]

                    if len(col_data) < n_new:
                        pad_size = n_new - len(col_data)
                        if c == "fold":
                            col_data += [-1] * pad_size
                        else:
                            col_data += ["synthetic"] * pad_size

                    df_train_new[c] = col_data
            
            # Prepare the validation data (DEMV is never applied to it)
            df_val_features = df_val_pre.drop(
                columns=[target] + [c for c in non_feature_columns if c in df_val_pre.columns]
            )

            # Recombine: the classifier expects X_cv (train+val) plus fixed_validation_mask
            X_train_pre = df_train_new.drop(columns=[target], errors='ignore')
            y_train_pre = df_train_new[target]

            # Align the columns between the post-DEMV training set and the validation set
            common_cols = [c for c in X_train_pre.columns if c in df_val_features.columns]
            X_cv = pd.concat([X_train_pre[common_cols], df_val_features[common_cols]], axis=0, ignore_index=True)
            y_cv = pd.concat([y_train_pre, df_val_pre[target]], axis=0, ignore_index=True)

            # Recompute fixed_validation_mask: training is the first N rows, validation the last ones
            n_train = len(X_train_pre)
            n_val = len(df_val_features)
            fixed_validation_mask = pd.Series(
                [False] * n_train + [True] * n_val,
                index=X_cv.index
            )

            # Stratification
            y_stratify_train_new = df_train_new[
                [c for c in y_stratify_keys if c in df_train_new.columns]
            ].apply(lambda x: '_'.join(x.astype(str)), axis=1)
            y_stratify_val_new = df_val_pre[y_stratify_keys].apply(lambda x: '_'.join(x.astype(str)), axis=1)
            stratify_cv = pd.concat([y_stratify_train_new, y_stratify_val_new], axis=0, ignore_index=True)

        else:
            # For None and Pos: no DEMV, df_data is used directly
            # The sensitive variables are passed along: the classifier drops them
            # from the predictors itself, but still needs them for the fairness metrics
            y_stratify = df_data[y_stratify_keys].apply(lambda x: '_'.join(x.astype(str)), axis=1)

            X_cv = df_data.drop(columns=[dict_['target']] + [c for c in non_feature_columns if c in df_data.columns])
            y_cv = df_data[dict_['target']]
            stratify_cv = y_stratify
            fixed_validation_mask = df_data["fold"].astype(int) == int(_validation_fold)

        # Load the already separated test data
        y_stratify_test = df_data_test[y_stratify_keys].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        X_test = df_data_test.drop(columns=[dict_['target']] + [c for c in non_feature_columns if c in df_data_test.columns])
        y_test = df_data_test[dict_['target']]
        stratify_test = y_stratify_test

        # Percentage distribution of each dataset split
        cv_dist = stratify_cv.value_counts(normalize=True).sort_index()
        test_dist = stratify_test.value_counts(normalize=True).sort_index()

        # Combine the distributions into a single DataFrame for comparison
        if _verbose:
            print(f"Cross-Validation set size: {X_cv.shape}")
            print(f"Final test set size: {X_test.shape}\n")
            comparison_df = pd.DataFrame({
                "Cross-Validation (%)": cv_dist * 100,
                "Test (%)": test_dist * 100
            })

            # Fill with 0 the rare case of a group missing from one of the splits
            comparison_df.fillna(0, inplace=True)

            # Comparison table
            print("Group percentage distribution comparison table:")
            print(comparison_df.round(2))
            print(f"Fixed validation fold: {_validation_fold}")

            print("="*80)
            print(f"Running {classify_type.upper()} for classification")
            print("="*80)
        
        if classify_type == "mlp":
            train_mlp(_dataset_name, X_cv, y_cv, 
                    X_test, y_test, stratify_cv, 
                    sensitive_features, _set_loss, mitigation_tech=_mitigation_tech,
                    opt_type=optimizer_mlp, batch_size=batch_size_mlp, k_folds=_k_folds, 
                    _epochs=epochs_mlp,
                    fixed_validation_mask=fixed_validation_mask,
                    validation_fold_value=_validation_fold)
        elif classify_type == "knn":
            train_knn(_dataset_name, X_cv, y_cv, X_test, y_test, stratify_cv, 
                    sensitive_features, mitigation_tech=_mitigation_tech,
                    fixed_validation_mask=fixed_validation_mask,
                    validation_fold_value=_validation_fold)
        elif classify_type == "dtree":
            train_dtree(_dataset_name, X_cv, y_cv, X_test, y_test, stratify_cv, 
                    sensitive_features, mitigation_tech=_mitigation_tech, max_depth=3,
                    fixed_validation_mask=fixed_validation_mask,
                    validation_fold_value=_validation_fold)


    else:
        raise Exception(f"Sorry, mitigation tech: {_mitigation_tech} not recognized")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Bias Mitigation on Fairness and Accuracy in Automated Skin Lesion Classification"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["db-pad-ufes-20", "db-hiba", "db-midas"],
        help="Name of the dataset to process"
    )

    parser.add_argument(
        "--mitigation",
        type=str,
        required=True,
        choices=["None", "Pre", "In", "PI", "PP", "IP", "Pos", "PIP"],
        help="Name of the bias mitigation technique"
    )

    parser.add_argument(
        "--classify",
        type=str,
        choices=["mlp", "dtree", "knn"],
        default="mlp",
        help="Name of the classifier to use"
    )

    parser.add_argument(
        "--num_epochs_vae",
        type=int,
        default= 2, #20000,
        help="Number of epochs used to train the VAE"
    )

    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Flag enabling the detailed output during the run"
    )

    parser.add_argument(
        "--validation_fold",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Fixed fold held out for validation in this run"
    )


    args = parser.parse_args()
    _dataset_name = args.dataset
    _mitigation_tech = args.mitigation
    _classify_type = args.classify
    _num_epochs_vae = args.num_epochs_vae
    _verbose = args.verbose
    _validation_fold = args.validation_fold
    
    _early_stop_patience = 50

    main(_dataset_name, _num_epochs_vae=_num_epochs_vae, _early_stop_patience=_early_stop_patience,
         _mitigation_tech=_mitigation_tech, classify_type=_classify_type, 
            _type_adv="vae", _verbose=_verbose, _validation_fold=_validation_fold)
            
 
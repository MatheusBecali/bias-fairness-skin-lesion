# -*- coding: utf-8 -*-
"""
Shared helper functions used across the experiments.

Groups the dataset feature definitions, the data-loading and preprocessing
utilities, the CSV writer for the results, and the routines that evaluate how
much sensitive information leaks into the learned latent space.

Author: Matheus Becali Rocha
Email: matheusbecali@gmail.com
"""

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

from src.vae import reparameterize


# Preprocessing and data-loading functions
def features_setting(data):
    """
    Define the feature groups of a dataset.

    Args:
        data: Dataset identifier ('adult', 'db-pad-ufes-20', 'db-hiba' or 'db-midas').

    Returns:
        A dict describing the dataset columns:
        - 'categorical_features': text columns that need label encoding
        - 'continuous_features': numeric columns
        - 'discrete_features': binary / one-hot / numeric-categorical columns
        - 'sensitive_features': protected attributes used by the fairness metrics
        - 'full_features': every feature column
        - 'normal_features': full_features minus the sensitive ones
        - 'target': name of the label column
    """
    dict_ = {}
    if data == "adult":
        dict_['categorical_features'] = ['marital_status', 'occupation', 'race', 'gender', 'workclass', 'education']
        dict_['continuous_features']  = ['age', 'hours_per_week']
        # dict_['sensitive_features'] = ['gender', 'age', 'race']
        dict_['sensitive_features'] = ['gender', 'race']
        dict_['target'] = 'income'
        dict_['full_features'] = dict_['categorical_features'] + dict_['continuous_features']
        dict_['normal_features'] = [x for x in dict_['full_features'] if x not in dict_['sensitive_features']]
        dict_['discrete_features'] = []
    elif data == "db-pad-ufes-20":
        dict_['categorical_features'] = [ ]

        dict_['continuous_features'] = [
            "age", "diameter_1", "diameter_2"
        ]

        dict_['discrete_features'] = [
            "img_info_benign", "img_info_malignant",  "gender", "fitzpatrick",	

            # symptoms reported by the patient (one-hot, UNK = unknown)
            "itch_True", "itch_False", "itch_UNK",
            "grew_True", "grew_UNK", "grew_False",
            "hurt_False", "hurt_True", "hurt_UNK",
            "changed_True", "changed_False", "changed_UNK",
            "bleed_True", "bleed_False", "bleed_UNK",
            "elevation_True", "elevation_False", "elevation_UNK",
        ]

        dict_['sensitive_features'] = ['gender', 'fitzpatrick']

        dict_['full_features'] = dict_['categorical_features'] + \
                                dict_['continuous_features'] + \
                                dict_['discrete_features']

        dict_['normal_features'] = [x for x in dict_['full_features'] if x not in dict_['sensitive_features']]

        dict_['target'] = 'diagnosis'  # or 'REAL' when the ground-truth label is used
    elif data == "db-hiba":
        dict_['categorical_features'] = [ ]

        dict_['continuous_features'] = [
            "age"
        ]


        dict_['discrete_features'] = [
            "img_info_benign", "img_info_malignant", "gender", "fitzpatrick",
            # "image_information", "gender",

            # General anatomical site of the lesion (one-hot)
            "anatom_site_general_anterior_torso",
            "anatom_site_general_upper_extremity", 
            "anatom_site_general_posterior_torso", 
            "anatom_site_general_lower_extremity", 
            "anatom_site_general_lateral_torso", 
            "anatom_site_general_head_neck", 
            "anatom_site_general_oral_genital", 
            "anatom_site_general_palms_soles",
            
            # Family and personal history of melanoma (one-hot)
            "family_hx_mm_False",
            "family_hx_mm_True", 
            "personal_hx_mm_False", 
            "personal_hx_mm_True"
        ]

        # dict_['sensitive_features'] = ['gender', 'fitzpatrick']
        dict_['sensitive_features'] = ['gender']

        dict_['full_features'] = dict_['categorical_features'] + \
                                dict_['continuous_features'] + \
                                dict_['discrete_features']

        dict_['normal_features'] = [x for x in dict_['full_features'] if x not in dict_['sensitive_features']]

        dict_['target'] = 'diagnosis'  # or 'REAL' when the ground-truth label is used
    elif data == "db-midas":
        dict_ = {}

        # No text categorical feature
        dict_['categorical_features'] = []

        # Continuous variables
        dict_['continuous_features'] = [
            "age",
            "length_(mm)",
            "width_(mm)"
        ]

        # Class probabilities predicted by the image model
        dict_['prob'] = ["img_info_benign", "img_info_malignant"]

        # Discrete variables (binary, one-hot or numeric categorical)
        dict_['discrete_features'] = dict_['prob'] + [
            "gender",
            "fitzpatrick",
            "midas_location_chest",
            "midas_location_left_lower_back",
            "midas_location_right_upper_eyelid",
            "midas_location_left_upper_back",
            "midas_location_right_dorsal_hand",
            "midas_location_nasal_bridge",
            "midas_location_right_forehead",
            "midas_location_right_posterior_shoulder",
            "midas_location_right_flank",
            "midas_location_left_cheek",
            "midas_location_right_upper_back",
            "midas_location_right_post_auricular_scalp",
            "midas_location_right_distal_lateral_upper_arm",
            "midas_location_posterior_midline_neck",
            "midas_location_left_proximal_dorsal_forearm",
            "midas_location_right_chest",
            "midas_location_left_lateral_neck",
            "midas_location_right_upper_arm",
            "midas_location_left_posterior_neck",
            "midas_location_left_dorsal_hand",
            "midas_location_right_posterior_helix",
            "midas_location_umbilicus",
            "midas_location_right_nasal_ala",
            "midas_location_left_antihelix",
            "midas_location_right_medial_eyebrow",
            "midas_location_left_forearm",
            "midas_location_left_lateral_thigh",
            "midas_location_left_nasal_ala",
            "midas_location_left_forehead",
            "midas_location_left_medial_ankle",
            "midas_location_right_medial_forearm",
            "midas_location_mid_back",
            "midas_location_right_mid_back",
            "midas_location_left_flank",
            "midas_location_chin",
            "midas_location_right_elbow",
            "midas_location_left_dorsal_2nd_toe",
            "midas_location_left_chest",
            "midas_location_left_elbow",
            "midas_location_right_abdomen",
            "midas_location_right_posterior_calf",
            "midas_location_left_anterior_thigh",
            "midas_location_left_medial_thigh",
            "midas_location_right_upper_shoulder",
            "midas_location_right_shoulder",
            "midas_location_right_lateral_calf",
            "midas_location_right_index_finger",
            "midas_location_left_upper_chest",
            "midas_location_right_lower_eyelid",
            "midas_location_mid_chest",
            "midas_location_right_lateral_neck",
            "midas_location_posterior_neck",
            "midas_location_frontal_scalp",
            "midas_location_right_medial_thigh",
            "midas_location_left_upper_arm",
            "midas_location_right_preauricular",
            "midas_location_left_thigh",
            "midas_location_right_4th_dorsal_toe",
            "midas_location_left_posterior_helix",
            "midas_location_nasal_tip",
            "midas_location_left_mandible",
            "midas_location_right_nasal_sidewall",
            "midas_location_left_lateral_calf",
            "midas_location_right_upper_post_calf",
            "midas_location_left_neck",
            "midas_location_right_forearm",
            "midas_location_right_medial_cheek",
            "midas_location_right_lower_back",
            "midas_location_left_lower_abdomen",
            "midas_location_left_anterior_neck",
            "midas_location_right_medial_shin",
            "midas_location_right_posterior_arm",
            "midas_location_right_axilla",
            "midas_location_mid_upper_back",
            "midas_location_right_thigh",
            "midas_location_right_clavicle",
            "midas_location_mid_low_back",
            "midas_location_right_posterior_thigh",
            "midas_location_right_shin",
            "midas_location_right_frontal_scalp",
            "midas_location_central_chest",
            "midas_location_right_cheek",
            "midas_location_left_shin",
            "midas_location_left_back",
            "midas_location_mid_lower_back",
            "midas_location_left_anterior_lower_leg",
            "midas_location_left_shoulder_lateral",
            "midas_location_left_mid_back",
            "midas_location_right_back",
            "midas_location_mid_abdomen",
            "midas_location_left_shoulder",
            "midas_location_left_foot",
            "midas_location_left_chest_medial",
            "midas_location_left_chest_lateral",
            "midas_location_left_abdomen",
            "midas_location_left_cutaneous_lip",
            "midas_location_right_leg",
            "midas_location_left_calf",
            "midas_location_left_malar_cheek",
            "midas_location_left_heel",
            "midas_location_right_lower_leg",
            "midas_location_right_3rd_finger",
            "midas_location_right_superior_helix_of_ear",
            "midas_location_right_posterior_neck",
            "midas_location_left_lobule_of_ear",
            "midas_location_right_calf",
            "midas_location_left_nasal_bridge",
            "midas_location_left_temple",
            "midas_location_left_postauricular",
            "midas_location_right_temple",
            "midas_location_right_deltoid",
            "midas_location_left_lower_lip",
            "midas_location_left_lower_vermilion_border_of_lip",
            "midas_location_abdomen",
            "midas_location_right_lateral_leg",
            "midas_location_left_posterior_shoulder",
            "midas_location_left_crown_of_scalp",
            "midas_location_right_popliteal_fossa",
            "midas_location_crown_of_scalp",
            "midas_location_right_retro_auricular",
            "midas_location_left_deltoid",
            "midas_location_right_neck",
            "midas_location_right_anterior_shin",
            "midas_location_left_upper_lip",
            "midas_location_nasal_dorsum",
            "midas_location_right_foot",
            "midas_location_left_nasal_tip",
            "midas_location_left_nose",
            "midas_location_shoulder",
            "midas_location_left_breast",
            "midas_location_right_lower_lateral_leg",
            "midas_location_left_upper_calf",
            "midas_location_left_lower_calf",
            "midas_location_central_lower_back",
            "midas_location_central_upper_back",
            "midas_location_left_melolabia_fold",
            "midas_location_left_lateral_cheek",
            "midas_location_left_lower_medial_cheek",
            "midas_location_right_crown_of_scalp",
            "midas_location_left_4th_finger",
            "midas_location_right_anterior_upper_arm",
            "midas_location_upper_mid_back",
            "midas_location_right_antihelix",
            "midas_location_left_posterior_calf",
            "midas_location_superior_l_upper_back",
            "midas_location_upper_chest",
            "midas_location_right_medial_arm",
            "midas_location_r4_digit",
            "midas_location_left_dorsal_foot",
            "midas_location_right_helix",
            "midas_location_mid_nasal_dorsum",
            "midas_location_left_parietal_scalp",
            "midas_location_upper_mid_r_chest",
            "midas_location_left_anterior_shin",
            "midas_location_left_lower_leg",
            "midas_location_upper_middle_lip",
            "midas_location_left_arm",
            "midas_location_left_preauricular",
            "midas_location_right_low_chest",
            "midas_location_left_knee",
            "midas_location_forehead",
            "midas_location_right_lateral_lower_leg",
            "midas_location_right_lower_leg_inferior",
            "midas_location_left_medial_cheek",
            "midas_location_right_vertex_scalp",
            "midas_location_left_infra_auricular",
            "midas_location_right_superior_helix",
            "midas_location_right_lateral_knee",
            "midas_location_right_infirmary_breast",
            "midas_location_left_alar_crease_of_nose",
            "midas_location_right_nasal_supra_tip",
            "midas_location_right_nasal_tip",
            "midas_location_left_ear",
            "midas_location_left_helix",
            "midas_location_left_frontal_scalp",
            "midas_location_right_inguinal_fold",
            "midas_location_left_buttock",
            "midas_location_right_medial_lower_leg",
            "midas_location_right_plantar_arch",
            "midas_location_right_mid_vertex",
            "midas_location_mid_central_back",
            "midas_location_left_central_back",
            "midas_location_left_wrist",
            "midas_location_right_chin",
            "midas_location_right_pre_auricular",
            "midas_location_left_alar_crease",
            "midas_location_philtrum",
            "midas_location_right_arm",
            "midas_location_left_root_of_neck",
            "midas_location_right_upper_abdomen",
            "midas_location_left_upper_thigh",
            "midas_location_right_superior_parietal",
            "midas_location_right_inferior_parietal",
            "midas_location_left_medial_shin",
            "midas_location_left_leg",
            "midas_location_left_clavicle",
            "midas_location_right_jaw",
            "midas_location_right_lateral_back",
            "midas_location_right_frontal_hairline",
            "midas_location_right_postauricular",
            "midas_location_left_axilla",
            "midas_location_left_superior_helix_of_ear",
            "midas_location_upper_back",
            "midas_location_left_vertex_scalp",
            "midas_location_left_alar_crease_(peri_nasal)",
            "midas_location_right_lateral_mid_back",
            "midas_location_left_medial_calf",
            "midas_location_right_preauricular_cheek",
            "midas_location_left_jawline",
            "midas_location_right_proximal_leg_melanoma_scar",
            "midas_location_left_upper_cutaneous_lip",
            "midas_location_left_dorsal_forearm",
            "midas_location_lower_abdomen",
            "midas_location_right_lateral_cheek",
            "midas_location_mid_vertex_scalp",
            "midas_location_left_hand",
            "midas_location_right_medial_chest",
            "midas_location_left_inner_upper_arm",
            "midas_location_right_lateral_ankle",
            "midas_location_mid_forehead",
            "midas_location_left_arm_medial",
            "midas_location_left_eyebrow",
            "midas_location_right_buttock",
            "midas_location_mid_upper_cutaneous_lip",
            "midas_location_mid_nasal_supra_tip",
            "midas_location_right_supra_nasal_tip",
            "midas_location_mid_posterior_neck",
            "midas_location_left_post_auricular",
            "midas_location_right_ankle",
            "midas_location_left_lower_flank",
            "midas_location_left_groin",
            "midas_location_right_lower_chest",
            "midas_location_right_nasal_fold",
            "midas_location_left_upper_vermilion_border",
            "midas_location_left_anterior_shoulder",
            "midas_location_mid_nasal_supratip",
            "midas_location_right_medial_knee",
            "midas_location_right_side_burn",
            "midas_location_left_thigh_distal",
            "midas_location_left_thigh_proximal",
            "midas_location_mid_vertex",
            "midas_location_right_conchal_bowl",
            "midas_location_right_upper_chest",
            "midas_location_upper_mid_abdomen",
            "midas_location_left_shoulder_anterior",
            "midas_location_left_shoulder_posterior",
            "midas_location_left_infra_mammary",
            "midas_location_right_lateral_thigh",
            "midas_location_right_anterior_leg",
            "midas_location_left_nasal_dorsum",
            "midas_location_left_scalp_vertex",
            "midas_location_left_medial_plantar_heel",
            "midas_location_left_mid_plantar_heel",
            "midas_location_right_base_of_neck",
            "midas_location_mons_pubis",
            "midas_location_left_suprapubic",
            "midas_location_central_upper_abdomen",
            "midas_location_left_nasal_supratip",
            "midas_location_left_lower_cutaneous_lip",
            "midas_location_right_distal_forearm",
            "midas_location_right_proximal_forearm",
            "midas_location_right_superior_upper_arm",
            "midas_location_left_lateral_jawline",
            "midas_location_left_nasal_sidewall",
            "midas_location_left_scapula",
            "midas_location_right_thumb_base",
            "midas_location_left_inguinal_crease",
            "midas_location_left_flank_superior",
            "midas_location_left_flank_inferior",
            "midas_location_lower_vermilion_lip",
            "midas_location_left_posterior_thigh",
            "midas_location_lower_back",
            "midas_location_right_nose_tip",
            "midas_location_right_superomedial_thigh",
            "midas_location_right_lateral_eyelid",
            "midas_location_left_posterior_ankle",
            "midas_location_right_lateral_foot",
            "midas_location_right_eyelid",
            "midas_location_mid_l_vertex_scalp",
            "midas_location_left_hip",
            "midas_location_left_superior_helix",
            "midas_location_left_inferior_upper_arm",
            "midas_location_left_lateral_knee",
            "midas_location_right_nasal_bridge",
            "midas_location_right_upper_lip",
            "midas_location_left_ear_scapha",
            "midas_location_left_lower_eyelid",
            "midas_location_right_dorsal_foot",
            "midas_location_right_nasal_supratip",
            "midas_location_left_hand_radial",
            "midas_location_left_medial_eyebrow",
            "midas_location_right_medial_malleolus",
            "midas_location_right_antitragus",
            "midas_location_right_pinky",
            "midas_location_right_lower_cheek",
            "midas_location_left_jaw",
            "midas_location_left_occiput",
        ]
        # Normalize the column names to match the CSV headers
        dict_['discrete_features'] = [c.strip().replace(" ", "_").lower() for c in dict_['discrete_features']]

        # Sensitive attributes (used by the fairness metrics)
        # dict_['sensitive_features'] = ['gender', 'fitzpatrick']
        dict_['sensitive_features'] = ['gender']

        # Combine every feature group
        dict_['full_features'] = (
            dict_['categorical_features']
            + dict_['continuous_features']
            + dict_['discrete_features']
        )

        # Regular (non-sensitive) features
        dict_['normal_features'] = [
            x for x in dict_['full_features']
            if x not in dict_['sensitive_features']
        ]

        # Target variable
        dict_['target'] = 'diagnosis'

    return dict_


def preprocess_dataset(df, categorical_features):
    """Encode the categorical features in place using a LabelEncoder."""
    for c in categorical_features:
        le = preprocessing.LabelEncoder()
        df[c] = le.fit_transform(df[c])
    return df

def prepare_data_loader(data, target, batch_size=32, shuffle=True, drop_last=False):
    """
    Wrap features and labels into a PyTorch DataLoader.

    Args:
        data: Feature tensor; cast to float32.
        target: Label tensor; cast to long, as expected by the loss functions.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the samples at every epoch.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        A tuple (dataloader, size), where `size` holds the shapes of the tensors.
    """
    # tensor_data = torch.tensor(np.array(data, dtype=np.float32))
    # tensor_target = torch.tensor(np.array(target), dtype=torch.long)
    tensor_data = data.to(torch.float32)
    tensor_target = target.to(torch.long)

    dataset = torch.utils.data.TensorDataset(tensor_data, tensor_target)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    size = {'x_size': tensor_data.size(),
            'y_size': tensor_target.size()}
    return dataloader, size

def save_results_to_csv(filename, results_data, header):
    """
    Append one result row to a CSV file, writing the header on first use.

    Args:
        filename: Path to the CSV file; opened in append mode.
        results_data: Dict holding the row, keyed by column name.
        header: Ordered list of column names.
    """
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csvfile:
        fieldnames = header
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Only write the header when the file is being created
        if not file_exists:
            writer.writeheader()

        writer.writerow(results_data)


def eval_adversary(encoder, adversary, data_loader, attribute_types, device, model_type="vae"):
    """
    Measure how well the adversary recovers the sensitive attributes from the latent space.

    A high accuracy (or a low MSE) means the encoder still leaks sensitive
    information, so the adversarial debiasing was not effective.

    Args:
        encoder: Trained encoder (VAE or AE).
        adversary: Adversarial head that predicts the sensitive attributes from z.
        data_loader: Loader yielding (x_batch, s_batch, _) tuples.
        attribute_types: One entry per sensitive attribute, 'binary' or 'regression'.
        device: Device on which to run inference.
        model_type: 'vae' (encoder returns mean/log_var) or 'ae' (returns z directly).

    Returns:
        A dict with 'Acc_attr{i}' for binary attributes and 'MSE_attr{i}' for
        regression ones.
    """
    encoder.eval()
    adversary.eval()
    all_true = [[] for _ in attribute_types]
    all_pred = [[] for _ in attribute_types]
    mse_list = [[] for _ in attribute_types]
    with torch.no_grad():
        for x_batch, s_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            s_batch = s_batch.to(device)
            if model_type == "vae":
                mean, log_var = encoder(x_batch)
                z = reparameterize(mean, log_var)
            elif model_type == "ae":
                z = encoder(x_batch)

            adv_preds = adversary(z)
            for i, (atype, preds) in enumerate(zip(attribute_types, adv_preds)):
                if atype == 'binary':
                    # Logits -> probabilities -> labels at the 0.5 threshold
                    probs = torch.sigmoid(preds).cpu().numpy()
                    pred_labels = (probs > 0.5).astype(int)
                    all_pred[i].extend(pred_labels.flatten())
                    # Fallback for a 1D s_batch (a single sensitive attribute)
                    try:
                        all_true[i].extend(s_batch[:, i].cpu().numpy().astype(int))
                    except:
                        all_true[i].extend(s_batch.cpu().numpy().astype(int))
                elif atype == 'regression':
                    pred_vals = preds.cpu().numpy().flatten()
                    # Fallback for a 1D s_batch (a single sensitive attribute)
                    try:
                        true_vals = s_batch[:, i].cpu().numpy().flatten()
                    except:
                        true_vals = s_batch.cpu().numpy().flatten()
                    all_pred[i].extend(pred_vals)
                    all_true[i].extend(true_vals)
                    mse = ((pred_vals - true_vals) ** 2)
                    mse_list[i].extend(mse)

    # Aggregate the batch-level predictions into one metric per attribute
    metrics = {}
    for i, atype in enumerate(attribute_types):
        if atype == 'binary':
            true = np.array(all_true[i])
            pred = np.array(all_pred[i])
            acc = (true == pred).mean() if len(true) > 0 else float('nan')
            metrics[f'Acc_attr{i}'] = acc
        elif atype == 'regression':
            mse = np.mean(mse_list[i]) if len(
                mse_list[i]) > 0 else float('nan')
            metrics[f'MSE_attr{i}'] = mse
    return metrics

def evaluate_fairness_latent(encoder, X, y_sensitive, device, dataset_name, mitigation_type, title="Fairness Evaluation", model_type="vae"):
    """
    Assess how much sensitive information remains in the latent space.

    Three complementary diagnostics are computed per sensitive attribute, plus a
    t-SNE plot saved under ./plots/{dataset_name}/. The closer the accuracy is
    to chance level and the closer mutual information and correlation are to
    zero, the more the representation is free of the sensitive attribute.

    Args:
        encoder: Trained encoder (VAE or AE).
        X: Feature matrix to project into the latent space.
        y_sensitive: Sensitive attributes, shaped [n_samples] or [n_samples, n_attrs].
        device: Device on which to run inference.
        dataset_name: Dataset name, used in the plot path.
        mitigation_type: Mitigation technique, used in the plot file name.
        title: Title printed on the t-SNE plot.
        model_type: 'vae' (encoder returns mean/log_var) or 'ae' (returns z directly).

    Returns:
        A dict keyed by 'attr_{i}' holding the accuracy, the mean mutual
        information and the mean absolute correlation of each attribute.
    """
    # 1. Extract the latent space (mean)

    os.makedirs(f'./plots/{dataset_name}', exist_ok=True)

    encoder.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        if model_type == "vae":
            mean, _ = encoder(X_tensor)
        elif model_type == "ae":
            mean = encoder(X_tensor)
            
        z_latent = mean.cpu().numpy()
    print(f"[INFO] Latent space extracted: {z_latent.shape}")

    # Reshape y_sensitive to 2D so a single attribute is handled like many
    if len(y_sensitive.shape) == 1:
        y_sensitive = y_sensitive.reshape(-1, 1)

    results = {}

    for i in range(y_sensitive.shape[1]):
        y_attr = y_sensitive[:, i]
        print(f"\n--- Evaluating sensitive attribute {i+1} ---")

        # 2. Predict the sensitive attribute (accuracy via cross-validation).
        # cross_val_score avoids the data leakage of fitting and predicting on
        # the same samples; accuracy near chance level means little leakage.
        from sklearn.model_selection import cross_val_score
        clf = LogisticRegression(max_iter=500)
        cv_scores = cross_val_score(clf, z_latent, y_attr, cv=5, scoring='accuracy')
        acc = cv_scores.mean()
        print(f"CV accuracy predicting sensitive attribute {i+1}: {acc:.3f} (+/-{cv_scores.std():.3f})")

        # 3. Mutual information between each latent dimension and the attribute
        mi = mutual_info_classif(z_latent, y_attr)
        print(f"Mean mutual information: {mi.mean():.3f}")

        # 4. Mean absolute linear correlation across the latent dimensions
        correlations = [np.corrcoef(z_latent[:,j], y_attr)[0,1] for j in range(z_latent.shape[1])]
        print(f"Mean absolute correlation: {np.mean(np.abs(correlations)):.3f}")

        # 5. t-SNE visualization; wrapped in try/except so a plotting failure
        # never aborts the evaluation
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            z_embedded = tsne.fit_transform(z_latent)
            plt.figure(figsize=(6, 4))
            scatter = plt.scatter(z_embedded[:,0], z_embedded[:,1], c=y_attr, cmap="coolwarm", alpha=0.6)
            plt.title(f't-SNE of the latent space colored by sensitive attribute {i+1}\n({title})')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.colorbar(scatter, label='Sensitive attribute')
            # plt.show()
            plt.savefig(f"./plots/{dataset_name}/{dataset_name}_{model_type}_{mitigation_type}_tsne.pdf")
        except Exception as e:
            print("Error while plotting t-SNE:", e)

        # Store the results of this attribute
        results[f"attr_{i+1}"] = {
            "acc_sensitive": acc,
            "mutual_info_mean": mi.mean(),
            "corr_abs_mean": np.mean(np.abs(correlations))
        }

    return results


def calculate_class_weights(sensitive_attributes_train, device, verbose=False):
    """
    Compute a class weight for each sensitive attribute from its class imbalance.

    Produces the 'pos_weight' of a binary classification problem, meant to be
    passed to PyTorch's F.binary_cross_entropy_with_logits loss.

    Args:
        sensitive_attributes_train (np.array): 2D NumPy array holding the
                                               sensitive attributes of the
                                               training set. Shape:
                                               [n_samples, n_attributes].
        device (torch.device): Device (CPU or CUDA) the weight tensors are sent to.
        verbose (bool): Whether to print the per-attribute class counts.

    Returns:
        list: A list of PyTorch tensors, each holding the weight computed for
              the positive class of the corresponding attribute.
    """
    if verbose:
        print("--- Computing the adversary class weights ---")

    pos_weights = []
    # Number of attributes, taken from the array shape
    num_sensitive_attributes = sensitive_attributes_train.shape[1]

    for i in range(num_sensitive_attributes):
        # Isolate the column of the current attribute
        attribute_column = sensitive_attributes_train[:, i]

        # Count the samples of each class (0 and 1)
        counts = np.bincount(attribute_column.astype(int))

        # Handle a class that never appears (unlikely, but safe)
        if len(counts) < 2:
            # With a single class there is no imbalance, so the weight is 1.0
            weight = 1.0
            count_0, count_1 = counts[0], 0
        else:
            count_0, count_1 = counts[0], counts[1]
            # pos_weight formula: (number of negatives / number of positives)
            if count_1 == 0:  # Avoid a division by zero
                weight = 1.0
            else:
                weight = count_0 / count_1

        # Append the weight as a PyTorch tensor on the right device
        pos_weights.append(torch.tensor([weight], device=device))
        if verbose:
            print(
                f"Attribute {i}: Class 0 ({count_0} samples), Class 1 ({count_1} samples) -> Computed weight: {weight:.2f}")
    if verbose:
        print("\nWeight list to be used during training:", pos_weights)
    return pos_weights

def load_best_hyperparameters(json_path):
    """
    Load the best hyperparameters found by the Optuna search.

    Args:
        json_path: Path to the JSON file holding a 'best_params' object.

    Returns:
        A tuple with the VAE parameters (lr_vae, lr_adv, lambda_adv, beta,
        batch_size_vae, optimizer_vae) followed by the MLP ones
        (batch_size_mlp, optimizer_mlp, epochs_mlp). Any key missing from the
        file comes back as None.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    best_params = data.get("best_params", {})

    # Extract the relevant parameters
    lr_vae = best_params.get("lr_vae")
    lr_adv = best_params.get("lr_adv")
    lambda_adv = best_params.get("lambda_adv")
    beta = best_params.get("beta")
    batch_size_vae = best_params.get("batch_size_vae")
    optimizer_vae = best_params.get("optimizer_vae")
    # MLP
    batch_size_mlp = best_params.get("batch_size_mlp")
    optimizer_mlp = best_params.get("optimizer_mlp")
    epochs_mlp = best_params.get("epochs_mlp")

    return lr_vae, lr_adv, lambda_adv, beta, batch_size_vae, optimizer_vae, batch_size_mlp, optimizer_mlp, epochs_mlp

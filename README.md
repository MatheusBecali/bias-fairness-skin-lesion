# The Impact of Bias Mitigation on Fairness and Accuracy in Automated Skin Lesion Classification

Official code for the paper *"The Impact of Bias Mitigation on Fairness and Accuracy in Automated Skin Lesion Classification"*.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

> **Status:** Under review at *Journal of Healthcare Informatics Research*. The paper link and DOI will be added here once published.

## Overview

This repository contains the implementation of a comprehensive bias mitigation pipeline for automated skin lesion classification. The project investigates how various debiasing techniques (pre-processing, in-processing, and post-processing) impact both fairness metrics and classification accuracy across different datasets and machine learning models.

### Key Contributions

- **Multi-stage debiasing framework**: Combines pre-processing (DEMV), in-processing (adversarial VAE), and post-processing (MLDebiaser) techniques
- **Comprehensive evaluation**: Assesses fairness metrics (statistical parity, equalized odds, equal opportunity) alongside accuracy metrics
- **Multiple datasets**: Experiments on diverse skin lesion classification datasets (DB-PAD, DB-HIBA, DB-MIDAS)
- **Flexible architecture**: Supports multiple classifiers (MLP, KNN, Decision Trees) and mitigation combinations

## Features

- **Bias Mitigation Techniques**:
  - Pre-processing: DEMV (Demographic Embedding Variation Matching)
  - In-processing: Adversarial Variational Autoencoder (VAE)
  - Post-processing: MLDebiaser
  - Combinable strategies: Pre, In, Post, Pre+In, Pre+Post, In+Post, Pre+In+Post

- **Performance Metrics**:
  - Accuracy, Balanced Accuracy
  - Precision, Recall, F1-Score
  - Per-fold and aggregated results

- **Fairness Metrics**:
  - Statistical Parity Difference
  - Disparate Impact Ratio
  - Equalized Odds Difference
  - Average Odds Difference

- **Cross-Validation**: Stratified k-fold validation with configurable fold selection

## Datasets

We investigated the performance of different bias mitigation stages using three fairness-oriented modified versions of this three datasets, resulting in the DB-PAD-UFES-20, DB-HIBA, and DB-MIDAS.

| Dataset | Name | Original |
|---------|------|---------|
| **DB-PAD-UFES-20** | PAD-UFES-20 | [Original data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) |
| **DB-HIBA** | Hospital Italiano Buenos Aires | [Original data](https://api.isic-archive.com/doi/hiba-skin-lesions/) | 
| **DB-MIDAS** | MRA-MIDAS: Multimodal Image Dataset for AI-based Skin Cancer | [Original data](https://aimi.stanford.edu/datasets/mra-midas-Multimodal-Image-Dataset-for-AI-based-Skin-Cancer) | 

## Requirements

- Python 3.10+
- PyTorch (with optional CUDA support)
- scikit-learn
- pandas
- numpy
- demv
- holisticai

For detailed dependencies, see `requirements.txt`.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MatheusBecali/bias-fairness-skin-lesion
cd bias-fairness-skin-lesion
```

2. Create a virtual environment (optional but recommended):
```bash
conda env create -f enviroments.yml
conda activate skin-lesion-env
```

Or with pip:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python main.py --help
```


## Project Structure

```
.
├── main.py                          # Main experiment pipeline
├── run_optuna.py                    # Hyperparameter optimization script
├── aggregate_results.py             # Aggregates results across runs
├── atopsis_analysis.py              # TOPSIS multi-criteria analysis
├── generate_decision_making_csv.py  # Generates decision matrices
├── resultsLatex.py                  # Generates LaTeX tables
├── requirements.txt                 # Python dependencies
├── enviroments.yml                  # Conda environment file
├── data/                            # Datasets (input)
│   ├── db-pad-ufes-20/
│   ├── db-hiba/
│   └── db-midas/
├── model/                           # Trained models (output)
├── results/                         # CSV results from experiments
├── debiased/                        # Debiased data from preprocessing
├── plots/                           # Generated visualization plots
├── latex/                           # Generated LaTeX tables
├── src/
│   ├── net.py                       # Neural network architectures
│   └── vae.py                       # VAE implementation
└── utils/
    └── helpers.py                   # Utility functions
```

## Outputs

Each experiment generates:

- **CSV Results**: Per-fold performance and fairness metrics in `results/`
- **Trained Models**: Serialized models in `model/`
- **Debiased Data**: Preprocessed/debiased datasets in `debiased/`
- **Visualizations**: Plots comparing fairness vs accuracy in `plots/`
- **LaTeX Tables**: Publication-ready tables in `latex/`

## Usage

### Basic Usage

Run a complete experiment with a specific dataset, mitigation technique, and classifier:

```bash
# Example 1: No mitigation, MLP classifier on DB-PAD-UFES-20
python main.py --dataset db-pad-ufes-20 --mitigation None --classify mlp

# Example 2: Full pipeline (Pre+In+Post), MLP classifier on DB-HIBA
python main.py --dataset db-hiba --mitigation PIP --classify mlp

# Example 3: Pre+Post mitigation, fold 3 validation on DB-MIDAS
python main.py --dataset db-midas --mitigation PP --classify mlp
```

### Command-line Arguments

```
--dataset          Dataset name (db-pad-ufes-20, db-hiba, db-midas)
--mitigation       Mitigation strategy (None, Pre, In, Pos, PI, PP, IP, PIP)
                   Where: P=Pre-processing, I=In-processing, Pos=Post-processing
--classify         Classifier (mlp, knn, tree)
--validation_fold  Specific fold number (optional, default: all folds)
--help             Show all available options
```

### Mitigation Strategies

| Code | Description | Stages |
|------|-------------|--------|
| `None` | No debiasing | - |
| `Pre` | Pre-processing only | DEMV |
| `In` | In-processing only | Adversarial VAE |
| `Pos` | Post-processing only | MLDebiaser |
| `PI` | Pre + In-processing | DEMV + Adversarial VAE |
| `PP` | Pre + Post-processing | DEMV + MLDebiaser |
| `IP` | In + Post-processing | Adversarial VAE + MLDebiaser |
| `PIP` | Full pipeline | DEMV + Adversarial VAE + MLDebiaser |


## Reproducibility

All experiments use a fixed random seed (`_seed = 78645`) to ensure reproducibility. CUDA device selection is automatic if available, falling back to CPU otherwise.

To ensure reproducibility:
1. Use the same Python and PyTorch versions (see `requirements.txt`)
2. Run on the same hardware or set explicit GPU/CPU configuration
3. Use provided hyperparameters or regenerate via `run_optuna.py`
4. Run the script `run_all.sh`

## Scripts

### `aggregate_results.py`
Aggregates individual fold results into summary statistics (mean, std, min, max).

```bash
python aggregate_results.py
```

### `atopsis_analysis.py`
Performs A-TOPSIS, an approach Based on TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) multi-criteria analysis to rank mitigation strategies.

```bash
python atopsis_analysis.py
```

### `generate_decision_making_csv.py`
Generates decision matrices for comparative analysis of different techniques.

```bash
python generate_decision_making_csv.py
```

### `resultsLatex.py`
Converts results to publication-ready LaTeX tables.

```bash
python resultsLatex.py
```

## Citation

If you use this code in your research, please cite:
<!--
```bibtex
 @article{becali2026bias,
  title={The Impact of Bias Mitigation on Fairness and Accuracy in Automated Skin Lesion Classification},
  author={Becali, Matheus and ...},
  journal={Journal of Healthcare Informatics Research},
  year={2026},
  note={Under review}
} 
```
-->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

If you wish to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## Contact

If you have any questions or feedback, feel free to reach out to me at [matheusbecali@gmail.com](matheusbecali@gmail.com).

## Acknowledgments

This work uses and extends the following libraries:
- [ATOSIS](https://github.com/paaatcha/decision-making): An approach Based on TOPSIS for Ranking Evolutionary Algorithms
- [DEMV](https://github.com/giordanoDaloisio/demv): Demographic Embedding Variation Matching
- [Holisticai](https://github.com/holistic-ai/holisticai): Fairness metrics and post-processing techniques

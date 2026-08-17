# -*- coding: utf-8 -*-

"""
Build LaTeX tables from the aggregated experiment results.

Reads the '*_aggregated.csv' files produced by aggregate_results.py (one per
classifier) and emits ready-to-include LaTeX tables reporting performance and
fairness metrics as mean +/- standard deviation, with the best cell of each
metric highlighted in bold.

Usage:
    python resultsLatex.py --dataset db-pad-ufes-20

Author: Matheus Becali Rocha
Email: matheusbecali@gmail.com
"""

import argparse
import os

import pandas as pd


def format_value(mean, std, bold=False):
    """
    Format a 'mean +/- standard deviation' pair as a LaTeX math cell.

    Args:
        mean: Mean value of the metric.
        std: Standard deviation of the metric.
        bold: Whether the cell should be highlighted as the best result.

    Returns:
        The LaTeX string for the cell, using a comma as the decimal separator.
    """
    formatted = f"${mean:.4f} \\pm {std:.4f}$"
    if bold:
        formatted = f"$\\textbf{{{mean:.4f} $\\pm$ {std:.4f}}}$"
    # Decimal point is swapped for a comma to match the brazilian formatting.
    return formatted.replace('.', ',')


def mitigation_sort_key(mitigation_name):
    """
    Sort mitigation techniques following the order used in the experiments.

    Unknown labels fall back to 99 so they are pushed to the end of the table,
    and the name itself is the tie-breaker.
    """
    order = {
        'None': 0,
        'Pre': 1,
        'In': 2,
        'PI': 3,
        'PP': 4,
        'IP': 5,
        'Pos': 6,
        'Post': 6,
        'PIP': 7,
    }
    return (order.get(mitigation_name, 99), str(mitigation_name))


def normalize_mitigation_name(name):
    """Standardize mitigation labels for printing ('Pos' is reported as 'Post')."""
    if str(name) == 'Pos':
        return 'Post'
    return str(name)


def extract_protected_attributes(reference_df):
    """
    Extract the protected attributes present in the fairness columns.

    Fairness columns are named like 'Statistical Parity (gender) - mean', so
    the attribute is the text between parentheses.
    """
    attrs = []
    for col in reference_df.columns:
        if 'Statistical Parity (' in col and ' - mean' in col:
            attr = col.split('(')[1].split(')')[0]
            if attr not in attrs:
                attrs.append(attr)
    return attrs


def build_best_lookup(rows, metrics):
    """
    Decide which cells to print in bold for each fairness metric.

    Disparate Impact is best when closest to 1.0; every other fairness metric
    is a difference, so it is best when closest to 0.0.

    Returns:
        A dict keyed by (mitigation, classifier, metric) holding True for the
        winning cell of each metric.
    """
    best_lookup = {}
    for metric in metrics:
        if 'Disparate Impact' in metric:
            best_row = min(rows, key=lambda r: abs(float(r[metric]) - 1.0))
        else:
            best_row = min(rows, key=lambda r: abs(float(r[metric])))

        best_lookup[(best_row['mitigation'], best_row['classifier'], metric)] = True

    return best_lookup


def build_best_lookup_performance(rows, metrics):
    """
    Decide which cells to print in bold for each performance metric.

    Performance metrics are all higher-is-better, so the maximum wins.
    """
    best_lookup = {}
    for metric in metrics:
        best_row = max(rows, key=lambda r: float(r[metric]))
        best_lookup[(best_row['mitigation'], best_row['classifier'], metric)] = True
    return best_lookup


def generate_multiclass_fairness_sidewaystable(dfs_by_classifier, dataset_name="db-pad-ufes-20"):
    """
    Build a sidewaystable of fairness metrics combining MLP/KNN/DT.

    Rows are grouped by mitigation technique (via \\multirow) and one block is
    emitted per protected attribute found in the results.
    """
    classifier_labels = {
        'mlp': 'MLP',
        'knn': 'KNN',
        'dtree': 'DT',
    }
    classifier_order = ['mlp', 'knn', 'dtree']

    # Any classifier works as reference: they all share the same column layout.
    reference_df = next(iter(dfs_by_classifier.values()))
    protected_attrs = extract_protected_attributes(reference_df)

    if not protected_attrs:
        return "% No fairness metric found in the aggregated files.\n"

    # Build the caption/label from the dataset name, e.g. 'db-pad-ufes-20'
    # becomes the caption 'PAD-UFES-20' and the label suffix 'padufes20'.
    dataset_caption = str(dataset_name).replace('db-', '').replace('-', ' ').upper().replace(' ', '-')
    label_dataset = str(dataset_name).replace('db-', '').replace('-', '')

    latex = "\\begin{sidewaystable}[p]\n"
    latex += "    \\centering\n"
    latex += f"    \\caption{{Average fairness metrics, {dataset_caption}.}}\n"
    latex += f"    \\label{{tab:{label_dataset}_fair}}\n"
    latex += "    \\begin{tabular}{@{}cccccc@{}}\n"
    latex += "    \\toprule\n"
    latex += "    \\textbf{Mitigation} & \\textbf{Classifier} & \\textbf{Statistical Parity} & \\textbf{Disparate Impact} & \\textbf{Equal Opportunity Diff} & \\textbf{Average Odds Diff} \\\\ \\midrule\n"

    for attr_idx, protected_attr in enumerate(protected_attrs):
        fairness_metrics = [
            f'Statistical Parity ({protected_attr})',
            f'Disparate Impact ({protected_attr})',
            f'Equal Opportunity Diff ({protected_attr})',
            f'Average Odds Diff ({protected_attr})',
        ]

        # Only keep metrics available for every classifier, so the block stays aligned.
        available_metrics = [m for m in fairness_metrics if all(f"{m} - mean" in df.columns for df in dfs_by_classifier.values())]
        if len(available_metrics) < 4:
            continue

        # Flatten every (classifier, mitigation) pair into a single list of cells.
        rows = []
        for clf_key, clf_df in dfs_by_classifier.items():
            for _, row in clf_df.iterrows():
                entry = {
                    'mitigation': normalize_mitigation_name(row['Mitigation Technic']),
                    'classifier': classifier_labels[clf_key],
                }
                for metric in available_metrics:
                    entry[metric] = row[f"{metric} - mean"]
                    entry[f"{metric} - std"] = row[f"{metric} - std"]
                rows.append(entry)

        best_lookup = build_best_lookup(rows, available_metrics)

        # Separate consecutive protected-attribute blocks with a rule.
        if attr_idx > 0:
            latex += "    \\midrule\n"
        latex += f"    \\multicolumn{{6}}{{c}}{{\\textbf{{{protected_attr.capitalize()}}}}} \\\\ \\midrule\n"

        mitigation_names = sorted({r['mitigation'] for r in rows}, key=mitigation_sort_key)

        for mit_idx, mitigation in enumerate(mitigation_names):
            # Group the classifiers of this mitigation and keep MLP/KNN/DT order.
            mitigation_rows = [r for r in rows if r['mitigation'] == mitigation]
            mitigation_rows = sorted(
                mitigation_rows,
                key=lambda r: classifier_order.index(next(k for k, v in classifier_labels.items() if v == r['classifier']))
            )

            if not mitigation_rows:
                continue

            for row_idx, row in enumerate(mitigation_rows):
                line = "    "
                # The mitigation name is printed once and spans its classifier rows.
                if row_idx == 0:
                    line += f"\\multirow{{{len(mitigation_rows)}}}*{{{mitigation}}}"
                else:
                    line += " "

                line += f" & {row['classifier']}"
                for metric in available_metrics:
                    bold = best_lookup.get((row['mitigation'], row['classifier'], metric), False)
                    line += f" & {format_value(row[metric], row[f'{metric} - std'], bold)}"

                line += " \\\\\n"
                latex += line

            if mit_idx < len(mitigation_names) - 1:
                latex += "    \\hline\n"

    latex += "    \\bottomrule\n"
    latex += "    \\end{tabular}\n"
    latex += "\\end{sidewaystable}\n"

    return latex


def generate_multiclass_performance_sidewaystable(dfs_by_classifier, dataset_name="db-pad-ufes-20"):
    """
    Build a sidewaystable of performance metrics combining MLP/KNN/DT.

    Rows are grouped by mitigation technique (via \\multirow); a classifier is
    skipped when any of the expected metric columns is missing.
    """
    classifier_labels = {
        'mlp': 'MLP',
        'knn': 'KNN',
        'dtree': 'DT',
    }
    classifier_order = ['mlp', 'knn', 'dtree']

    # CSV column name -> short header used in the LaTeX table.
    metric_map = {
        'Test - Accuracy Score': 'ACC',
        'Test - Balanced Accuracy Score': 'BACC',
        'Test - Precision Score': 'Precision',
        'Test - Recall Score': 'Recall',
        'Test - F1 Score': 'F1 Score',
    }

    dataset_caption = str(dataset_name).replace('db-', '').replace('-', ' ').upper().replace(' ', '-')
    label_dataset = str(dataset_name).replace('db-', '').replace('-', '')

    # Flatten every (classifier, mitigation) pair into a single list of cells.
    rows = []
    for clf_key, clf_df in dfs_by_classifier.items():
        for _, row in clf_df.iterrows():
            entry = {
                'mitigation': normalize_mitigation_name(row['Mitigation Technic']),
                'classifier': classifier_labels[clf_key],
            }

            # Drop the entry entirely if any metric column is absent.
            missing_metric = False
            for metric in metric_map.keys():
                mean_col = f"{metric} - mean"
                std_col = f"{metric} - std"
                if mean_col not in clf_df.columns or std_col not in clf_df.columns:
                    missing_metric = True
                    break
                entry[metric] = row[mean_col]
                entry[f"{metric} - std"] = row[std_col]

            if not missing_metric:
                rows.append(entry)

    if not rows:
        return "% No performance metric found in the aggregated files.\n"

    best_lookup = build_best_lookup_performance(rows, list(metric_map.keys()))

    latex = "\\begin{sidewaystable}[p]\n"
    latex += "    \\centering\n"
    latex += f"    \\caption{{Average performance metrics, {dataset_caption}.}}\n"
    latex += f"    \\label{{tab:{label_dataset}_perf}}\n"
    latex += "    \\begin{tabular}{@{}ccccccc@{}}\n"
    latex += "    \\toprule\n"
    latex += "    \\textbf{Mitigation} & \\textbf{Classifier} & \\textbf{ACC} & \\textbf{BACC} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1 Score} \\\\ \\midrule\n"

    mitigation_names = sorted({r['mitigation'] for r in rows}, key=mitigation_sort_key)
    for mit_idx, mitigation in enumerate(mitigation_names):
        # Group the classifiers of this mitigation and keep MLP/KNN/DT order.
        mitigation_rows = [r for r in rows if r['mitigation'] == mitigation]
        mitigation_rows = sorted(
            mitigation_rows,
            key=lambda r: classifier_order.index(next(k for k, v in classifier_labels.items() if v == r['classifier']))
        )

        if not mitigation_rows:
            continue

        for row_idx, row in enumerate(mitigation_rows):
            line = "    "
            # The mitigation name is printed once and spans its classifier rows.
            if row_idx == 0:
                line += f"\\multirow{{{len(mitigation_rows)}}}*{{{mitigation}}}"
            else:
                line += " "

            line += f" & {row['classifier']}"
            for metric in metric_map.keys():
                bold = best_lookup.get((row['mitigation'], row['classifier'], metric), False)
                line += f" & {format_value(row[metric], row[f'{metric} - std'], bold)}"

            line += " \\\\\n"
            latex += line

        if mit_idx < len(mitigation_names) - 1:
            latex += "    \\hline\n"

    latex += "    \\bottomrule\n"
    latex += "    \\end{tabular}\n"
    latex += "\\end{sidewaystable}\n"
    return latex

def find_best_values(df, metrics):
    """Locate the best value of each metric (higher is better for performance)."""
    best_indices = {}
    for metric in metrics:
        mean_col = f"{metric} - mean"
        if mean_col in df.columns:
            # For most performance metrics, higher is better
            best_idx = df[mean_col].idxmax()
            best_indices[metric] = best_idx
    return best_indices

def generate_performance_table(df, dataset_name="DB-PAD-UFES-20"):
    """Build the single-classifier performance table."""

    # Performance metrics: CSV column name -> short header
    metrics = {
        'Accuracy Score': 'ACC',
        'Balanced Accuracy Score': 'BACC',
        'Precision Score': 'Precision',
        'Recall Score': 'Recall',
        'F1 Score': 'F1 Score'
    }

    # Find the best value of each metric so it can be printed in bold
    best_indices = find_best_values(df, metrics.keys())

    # Table header
    latex = "\\begin{table}[H]\n"
    latex += "\\centering % Center the table\n"
    latex += "\\begin{adjustbox}{width=1\\textwidth} % Shrink the table to fit the page width\n"
    latex += "\\begin{tabular}{@{}cccccc@{}}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Mitigation} & " + " & ".join([f"\\textbf{{{v}}}" for v in metrics.values()]) + " \\\\ \\midrule\n"

    # Data rows: one per mitigation technique
    for idx, row in df.iterrows():
        mitigation = row['Mitigation Technic']
        line = f"{mitigation}"

        for metric, label in metrics.items():
            mean_col = f"{metric} - mean"
            std_col = f"{metric} - std"

            if mean_col in df.columns and std_col in df.columns:
                mean = row[mean_col]
                std = row[std_col]
                bold = (idx == best_indices.get(metric, -1))
                line += f" & {format_value(mean, std, bold)}"

        line += " \\\\\n"
        latex += line

    # Table footer
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{adjustbox}\n"
    latex += f"\\caption{{Average performance metrics, {dataset_name}.}}\n"
    latex += "\\label{tab:origData}\n"
    latex += "\\end{table}\n"

    return latex

def generate_fairness_table(df, dataset_name="DB-PAD-UFES-20", protected_attr="gender"):
    """Build the single-classifier fairness table for one protected attribute."""

    # Fairness metrics reported for the given protected attribute
    fairness_metrics = [
        f'Statistical Parity ({protected_attr})',
        f'Disparate Impact ({protected_attr})',
        f'Equal Opportunity Diff ({protected_attr})',
        f'Average Odds Diff ({protected_attr})',
    ]

    # Keep only the metrics actually present in the DataFrame
    available_metrics = [m for m in fairness_metrics if f"{m} - mean" in df.columns]

    if not available_metrics:
        return f"% No fairness metric found for {protected_attr}\n"

    # Column headers
    metric_labels = {
        f'Statistical Parity ({protected_attr})': 'Statistical Parity',
        f'Disparate Impact ({protected_attr})': 'Disparate Impact',
        f'Equal Opportunity Diff ({protected_attr})': 'Equal Opportunity Diff',
        f'Average Odds Diff ({protected_attr})': 'Average Odds Diff',
    }

    # Find the best value of each metric (for fairness, smaller disparity is better)
    best_indices = {}
    for metric in available_metrics:
        mean_col = f"{metric} - mean"
        # For Statistical Parity and the difference metrics, closer to 0 is better
        # For Disparate Impact, closer to 1 is better
        if 'Disparate Impact' in metric:
            best_idx = df[mean_col].apply(lambda x: abs(x - 1)).idxmin()
        else:
            best_idx = df[mean_col].abs().idxmin()
        best_indices[metric] = best_idx

    # Table header
    latex = "\\begin{table}[H]\n"
    latex += "\\centering % Center the table\n"
    latex += "\\begin{adjustbox}{width=1\\textwidth} % Shrink the table to fit the page width\n"
    latex += f"\\begin{{tabular}}{{@{{}}c{'c' * len(available_metrics)}@{{}}}}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Mitigation} & " + " & ".join([f"\\textbf{{{metric_labels[m]}}}" for m in available_metrics]) + " \\\\ \\midrule\n"
    latex += f"\\multicolumn{{{len(available_metrics) + 1}}}{{c}}{{\\textbf{{{protected_attr.capitalize()}}}}} \\\\ \\midrule\n"

    # Data rows: one per mitigation technique
    for idx, row in df.iterrows():
        mitigation = row['Mitigation Technic']
        line = f"{mitigation}"

        for metric in available_metrics:
            mean_col = f"{metric} - mean"
            std_col = f"{metric} - std"

            mean = row[mean_col]
            std = row[std_col]
            bold = (idx == best_indices.get(metric, -1))
            line += f" & {format_value(mean, std, bold)}"

        line += " \\\\\n"
        latex += line

    # Table footer
    latex += "\\midrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{adjustbox}\n"
    latex += f"\\caption{{Average fairness metrics, {dataset_name}.}}\n"
    latex += "\\end{table}\n"

    return latex

def process_multiclass_csv(dataset_name="db-pad-ufes-20"):
    """
    Read the aggregated CSVs of MLP/KNN/DT and write the combined LaTeX tables.

    Both tables are printed to stdout and saved under ./latex/.
    """
    classifier_paths = {
        'mlp': f"./results/classification_model/mlp/{dataset_name}_aggregated.csv",
        'knn': f"./results/classification_model/knn/{dataset_name}_aggregated.csv",
        'dtree': f"./results/classification_model/dtree/{dataset_name}_aggregated.csv",
    }

    # keep_default_na=False preserves the literal "None" of the baseline runs.
    dfs_by_classifier = {}
    for clf, csv_path in classifier_paths.items():
        df = pd.read_csv(csv_path, keep_default_na=False)
        if df.empty:
            print(f"WARNING: No record found for {clf}")
            return
        dfs_by_classifier[clf] = df

    fairness_table = generate_multiclass_fairness_sidewaystable(dfs_by_classifier, dataset_name)
    performance_table = generate_multiclass_performance_sidewaystable(dfs_by_classifier, dataset_name)

    print("=" * 80)
    print("MULTI-CLASSIFIER PERFORMANCE TABLE")
    print("=" * 80)
    print(performance_table)

    print("=" * 80)
    print("MULTI-CLASSIFIER FAIRNESS TABLE")
    print("=" * 80)
    print(fairness_table)

    output_file_perf = f"./latex/latex_tables_{dataset_name}_perf_multiclassifier.tex"
    os.makedirs(os.path.dirname(output_file_perf), exist_ok=True)
    output_file_fair = f"./latex/latex_tables_{dataset_name}_multiclassifier.tex"
    os.makedirs(os.path.dirname(output_file_fair), exist_ok=True)
    with open(output_file_perf, 'w', encoding='utf-8') as f:
        f.write(performance_table)
    with open(output_file_fair, 'w', encoding='utf-8') as f:
        f.write(fairness_table)

    print("\n" + "=" * 80)
    print(f"Performance table saved to '{output_file_perf}'")
    print(f"Fairness table saved to '{output_file_fair}'")
    print("=" * 80)

# Usage example
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process the experiment results and generate LaTeX tables.")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["db-pad-ufes-20", "db-hiba", "db-midas"],
        help="Name of the dataset to process"
    )

    args = parser.parse_args()
    _dataset_name = args.dataset

    process_multiclass_csv(_dataset_name)

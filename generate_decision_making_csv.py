"""
Create decision-making CSV summaries (avg/std) from aggregated classifier results.

Reads the '*_aggregated.csv' files produced by aggregate_results.py (one per
classifier) and reshapes them into the two matrices expected by the
multi-criteria decision-making methods (TOPSIS / A-TOPSIS / TODIM): one holding
the mean of every metric and one holding its standard deviation. Each row is a
(classifier, mitigation) pair such as 'MLP (Pre)', and each column is a
criterion.

Usage:
    python generate_decision_making_csv.py
    python generate_decision_making_csv.py --results-root ./results --output-dir ./results/decision_making
"""

import argparse
from pathlib import Path

import pandas as pd

# Directory name of each classifier -> label printed in the output rows
CLASSIFIER_LABELS = {
    "mlp": "MLP",
    "knn": "KNN",
    "dtree": "DT",
}

# Column order of the performance criteria in the output CSVs
PERFORMANCE_METRICS_ORDER = [
    "Accuracy Score",
    "Balanced Accuracy Score",
    "Precision Score",
    "Recall Score",
    "F1 Score",
]

# Column order of the fairness criteria; repeated once per protected attribute
FAIRNESS_METRICS_BASE_ORDER = [
    "Statistical Parity",
    "Disparate Impact",
    "Equal Opportunity Diff",
    "Average Odds Diff",
]

def normalize_mitigation_name(name: str) -> str:
    """Normalize mitigation labels for final CSV output ('Pos' is written as 'Post')."""
    if str(name) == "Pos":
        return "Post"
    return str(name)


def mitigation_sort_key(mitigation_name: str):
    """
    Keep mitigation order stable and aligned with experiments.

    Unknown labels fall back to 99 so they land at the end, and the name itself
    breaks ties.
    """
    order = {
        "None": 0,
        "Pre": 1,
        "In": 2,
        "PI": 3,
        "PP": 4,
        "IP": 5,
        "Post": 6,
        "PIP": 7,
    }
    return (order.get(mitigation_name, 99), str(mitigation_name))


def canonical_metric_name(col_name: str) -> str:
    """
    Convert aggregated column names to output CSV metric names.

    Performance columns carry a 'Test - ' prefix in the aggregated files
    ('Test - F1 Score'), which is dropped here; fairness columns have no prefix
    and pass through unchanged.
    """
    if col_name.startswith("Test - "):
        return col_name[len("Test - ") :]
    return col_name


def collect_metric_columns(sample_df: pd.DataFrame):
    """
    Collect mean/std metric columns from an aggregated dataframe.

    Only metrics carrying both the ' - mean' and the ' - std' column are kept,
    since the decision-making methods need the pair.

    Returns:
        A dict mapping the metric name to its (mean_column, std_column) pair.
    """
    metric_pairs = {}
    for col in sample_df.columns:
        if not col.endswith(" - mean"):
            continue
        metric = col[: -len(" - mean")]
        std_col = f"{metric} - std"
        # Skip a mean with no matching std
        if std_col not in sample_df.columns:
            continue
        metric_pairs[metric] = (col, std_col)
    return metric_pairs


def ordered_metric_names(metric_pairs):
    """
    Order output metrics: performance first, then fairness grouped by attribute.

    Fairness metrics are recognized by the protected attribute in parentheses
    ('Statistical Parity (gender)'). Attributes keep the order in which they
    appear in the file, and inside each attribute the metrics follow
    FAIRNESS_METRICS_BASE_ORDER.

    Returns:
        The list of metric names in the order they should become CSV columns.
    """
    canonical_names = [canonical_metric_name(m) for m in metric_pairs.keys()]

    perf_names = [m for m in PERFORMANCE_METRICS_ORDER if m in canonical_names]

    fairness_names = [
        m for m in canonical_names if m not in perf_names and "(" in m and ")" in m
    ]

    # Collect the protected attributes, preserving first-seen order
    attr_order = []
    for metric in fairness_names:
        attr = metric.split("(")[-1].split(")")[0]
        if attr not in attr_order:
            attr_order.append(attr)

    # Group the fairness metrics by attribute, following the canonical order
    ordered_fairness = []
    for attr in attr_order:
        for base in FAIRNESS_METRICS_BASE_ORDER:
            candidate = f"{base} ({attr})"
            if candidate in fairness_names:
                ordered_fairness.append(candidate)

    return perf_names + ordered_fairness


def load_classifier_data(classification_root: Path, dataset: str):
    """
    Load all classifier aggregated CSVs for a given dataset.

    Missing classifiers are silently skipped, so the script still runs when only
    part of the experiments has finished.

    Returns:
        A dict mapping the classifier key to its aggregated DataFrame.
    """
    dfs = {}
    for clf_key in CLASSIFIER_LABELS:
        file_path = classification_root / clf_key / f"{dataset}_aggregated.csv"
        if not file_path.exists():
            continue
        # keep_default_na=False preserves the literal "None" of the baseline runs
        dfs[clf_key] = pd.read_csv(file_path, keep_default_na=False)
    return dfs


def build_avg_std_tables(dfs_by_classifier):
    """
    Build avg/std tables with rows like 'MLP (Pre)' and metric columns.

    Both tables share the exact same rows and columns: the decision-making
    methods read the means as the criteria matrix and the standard deviations as
    the associated uncertainty.

    Returns:
        A tuple (avg_df, std_df), or (None, None) when no classifier was loaded.
    """
    if not dfs_by_classifier:
        return None, None

    # Use the first dataframe as schema reference.
    sample_df = next(iter(dfs_by_classifier.values()))
    metric_pairs = collect_metric_columns(sample_df)
    metric_names = ordered_metric_names(metric_pairs)

    avg_rows = []
    std_rows = []

    # Fixed row order, independent of the dict iteration order
    classifier_order = ["mlp", "knn", "dtree"]

    for clf_key in classifier_order:
        if clf_key not in dfs_by_classifier:
            continue

        clf_label = CLASSIFIER_LABELS[clf_key]
        df = dfs_by_classifier[clf_key].copy()
        if "Mitigation Technic" not in df.columns:
            continue

        df["Mitigation Technic"] = df["Mitigation Technic"].apply(normalize_mitigation_name)
        df = df.sort_values(by="Mitigation Technic", key=lambda s: s.map(mitigation_sort_key))

        for _, row in df.iterrows():
            mitigation = row["Mitigation Technic"]
            # One alternative of the decision problem, e.g. 'MLP (Pre)'
            algorithm_name = f"{clf_label} ({mitigation})"

            avg_entry = {"Algorithms": algorithm_name}
            std_entry = {"Algorithms": algorithm_name}

            for metric_name in metric_names:
                # Rebuild the source column name: performance metrics carry the
                # 'Test - ' prefix that canonical_metric_name stripped away
                source_metric = metric_name
                if source_metric in PERFORMANCE_METRICS_ORDER:
                    source_metric = f"Test - {source_metric}"

                mean_col = f"{source_metric} - mean"
                std_col = f"{source_metric} - std"

                # pd.NA when the classifier lacks this metric, so both tables
                # keep the same shape
                avg_entry[metric_name] = row.get(mean_col, pd.NA)
                std_entry[metric_name] = row.get(std_col, pd.NA)

            avg_rows.append(avg_entry)
            std_rows.append(std_entry)

    avg_df = pd.DataFrame(avg_rows)
    std_df = pd.DataFrame(std_rows)

    # Enforce the column order and guarantee both tables are aligned
    ordered_columns = ["Algorithms"] + metric_names
    avg_df = avg_df.reindex(columns=ordered_columns)
    std_df = std_df.reindex(columns=ordered_columns)

    return avg_df, std_df


def save_dataset_tables(output_dir: Path, dataset: str, avg_df: pd.DataFrame, std_df: pd.DataFrame):
    """
    Save avg/std csv files for one dataset.

    Returns:
        A tuple (avg_path, std_path) with the paths actually written.
    """
    dataset_prefix = dataset
    avg_path = output_dir / f"{dataset_prefix}_avg_all.csv"
    std_path = output_dir / f"{dataset_prefix}_std_all.csv"

    avg_df.to_csv(avg_path, index=False)
    std_df.to_csv(std_path, index=False)

    return avg_path, std_path


def main():
    """Discover every aggregated dataset and write its avg/std tables."""
    parser = argparse.ArgumentParser(
        description="Generate decision-making avg/std CSVs from classification_model aggregated results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("./results"),
        help="Root folder containing classification_model and decision_making",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results/decision_making"),
        help="Output folder for generated avg/std CSVs",
    )

    args = parser.parse_args()

    classification_root = args.results_root / "classification_model"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover the datasets from the file names, deduplicating across classifiers
    dataset_names = sorted(
        {
            p.name.replace("_aggregated.csv", "")
            for p in classification_root.glob("*/*_aggregated.csv")
        }
    )

    if not dataset_names:
        print("No aggregated files found in classification_model.")
        return

    print("Datasets found:", ", ".join(dataset_names))

    for dataset in dataset_names:
        dfs_by_classifier = load_classifier_data(classification_root, dataset)
        avg_df, std_df = build_avg_std_tables(dfs_by_classifier)

        if avg_df is None or std_df is None or avg_df.empty or std_df.empty:
            print(f"[SKIP] {dataset}: no valid rows found")
            continue

        avg_path, std_path = save_dataset_tables(output_dir, dataset, avg_df, std_df)
        print(f"[OK] {dataset}")
        print(f"  avg -> {avg_path}")
        print(f"  std -> {std_path}")


if __name__ == "__main__":
    main()

"""
Aggregate per-fold results into mean +/- standard deviation.

Reads the '*_per_fold.csv' files produced by main.py and writes a summary
CSV holding the mean and the standard deviation of every metric, grouped by
(Dataset, Mitigation Technic).

Usage:
    python aggregate_results.py                        # every classifier and dataset
    python aggregate_results.py --classify mlp         # MLP only
    python aggregate_results.py --dataset db-pad-ufes-20 --classify mlp
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def aggregate_fold_results(input_csv: str, output_csv: str):
    """
    Read a per-fold results CSV and write a CSV holding mean +/- std.

    Args:
        input_csv: Path to the per-fold CSV produced by main.py.
        output_csv: Path where the aggregated CSV will be written.

    Returns:
        The aggregated DataFrame, or None when the input file is missing/empty.
    """
    if not os.path.exists(input_csv):
        print(f"File not found: {input_csv}")
        return

    # keep_default_na=False keeps the literal string "None" as text instead of NaN.
    df = pd.read_csv(input_csv, keep_default_na=False)

    # pandas may still read the text "None" as NaN, which would make groupby drop
    # the baseline (unmitigated) runs. Force those cases back to the "None" string.
    if 'Mitigation Technic' in df.columns:
        df['Mitigation Technic'] = (
            df['Mitigation Technic']
            .fillna('None')
            .astype(str)
            .str.strip()
            .replace({'': 'None'})
        )

    if df.empty:
        print(f"Empty file: {input_csv}")
        return

    # Grouping columns (non-numeric / metadata): one output row per combination.
    group_cols = ['Dataset', 'Mitigation Technic']
    # Columns that must not be averaged (identifiers and hyperparameters).
    skip_cols = {'Fold', '_batch_size', 'Loss Function'}

    # Every remaining numeric column is treated as a metric to aggregate.
    metric_cols = [
        c for c in df.columns
        if c not in group_cols and c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    # sort=False preserves the original ordering of the experiments;
    # dropna=False keeps groups whose key is missing.
    grouped = df.groupby(group_cols, sort=False, dropna=False)

    rows = []
    for name, group in grouped:
        row = {}
        # Metadata: `name` is a tuple when grouping by more than one column.
        if isinstance(name, tuple):
            for col, val in zip(group_cols, name):
                row[col] = val
        else:
            row[group_cols[0]] = name

        # Extra information that is constant within the group: take the first record.
        for col in skip_cols:
            if col in df.columns and col != 'Fold':
                row[col] = group[col].iloc[0]

        # Number of folds actually aggregated into this row.
        row['N_Folds'] = len(group)

        # Metrics: mean +/- std across the folds of this group.
        for col in metric_cols:
            values = group[col].dropna()
            if len(values) > 0:
                row[f'{col} - mean'] = np.mean(values)
                row[f'{col} - std'] = np.std(values)
            else:
                # No valid value for this metric in this group.
                row[f'{col} - mean'] = np.nan
                row[f'{col} - std'] = np.nan

        rows.append(row)

    df_agg = pd.DataFrame(rows)

    # Make sure the destination directory exists before writing.
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    df_agg.to_csv(output_csv, index=False)
    print(f"Aggregated results saved to: {output_csv}")
    print(f"{len(df_agg)} combinations (Dataset x Mitigation) from {len(df)} folds\n")

    return df_agg


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-fold results into mean +/- standard deviation"
    )
    parser.add_argument(
        "--classify",
        type=str,
        choices=["mlp", "knn", "dtree"],
        default=None,
        help="Classifier to aggregate (default: all)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["db-pad-ufes-20", "db-hiba", "db-midas"],
        default=None,
        help="Dataset to aggregate (default: all)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/classification_model",
        help="Root directory holding the results"
    )
    args = parser.parse_args()

    # When no filter is given, sweep over every classifier/dataset combination.
    classifiers = [args.classify] if args.classify else ["mlp", "knn", "dtree"]
    datasets = [args.dataset] if args.dataset else ["db-pad-ufes-20", "db-hiba", "db-midas"]

    print("=" * 70)
    print("Per-Fold Results Aggregation -> Mean +/- Standard Deviation")
    print("=" * 70)

    for clf in classifiers:
        for ds in datasets:
            input_csv = os.path.join(args.results_dir, clf, f"{ds}_per_fold.csv")
            output_csv = os.path.join(args.results_dir, clf, f"{ds}_aggregated.csv")

            if os.path.exists(input_csv):
                print(f"\nProcessing: {clf.upper()} / {ds}")
                aggregate_fold_results(input_csv, output_csv)
            else:
                # Fall back to a glob search: file names may carry extra suffixes.
                pattern = os.path.join(args.results_dir, clf, f"*{ds}*per_fold*")
                matches = glob.glob(pattern)
                if matches:
                    for match in matches:
                        basename = os.path.basename(match).replace('_per_fold', '_aggregated')
                        out = os.path.join(os.path.dirname(match), basename)
                        print(f"\nProcessing: {clf.upper()} / {os.path.basename(match)}")
                        aggregate_fold_results(match, out)

    print("\n" + "=" * 70)
    print("Aggregation finished!")
    print("=" * 70)


if __name__ == "__main__":
    main()

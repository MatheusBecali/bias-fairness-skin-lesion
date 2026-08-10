#!/bin/bash

datasets=("db-pad-ufes-20" "db-hiba" "db-midas")

for dataset in "${datasets[@]}"; do
    echo ">>> Running: dataset=$dataset"
    python resultsLatex.py --dataset "$dataset"
done


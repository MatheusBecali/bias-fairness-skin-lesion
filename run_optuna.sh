#!/bin/bash

datasets=("db-pad-ufes-20" "db-hiba" "db-midas")

for dataset in "${datasets[@]}"; do

    echo ">>> Running Optuna: dataset=$dataset"
    python run_optuna.py --dataset "$dataset"

done

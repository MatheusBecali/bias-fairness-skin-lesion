#!/bin/bash

classifiers=("mlp" "dtree" "knn")
datasets=("db-pad-ufes-20" "db-hiba" "db-midas")  
mitigations=("None" "Pre" "In" "PI" "PP" "IP" "Pos" "PIP")
folds=(1 2 3 4 5)

for classifier in "${classifiers[@]}"; do
    for dataset in "${datasets[@]}"; do
        for mitigation in "${mitigations[@]}"; do
            for fold in "${folds[@]}"; do

                echo ">>> Running: dataset=$dataset  mitigation=$mitigation  classify=$classifier  fold=$fold"
                python main.py --dataset "$dataset" --mitigation "$mitigation" --classify "$classifier" --validation_fold "$fold"

            done
        done
    done
done

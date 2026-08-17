# -*- coding: utf-8 -*-
"""
DESCRIPTION:

Code to analyze the results of the decision making process using ATOPSIS and/or TOPSIS methods.
The code reads the average and standard deviation matrices from csv files, computes the rankings of
the algorithms based on the specified benchmarks, and plots the rankings.

The code is structured in two approaches:
    1. Using the ATOPSIS (Krohling and Pacheco, 2015) method, which takes into account both the
    average and standard deviation of the benchmarks.
    2. Using the TOPSIS (Hwang and Yoon, 1981) method, which considers only the average values of
    the benchmarks.

The input matrices are the ones produced by generate_decision_making_csv.py: each row is an
alternative, that is a (classifier, mitigation) pair such as 'MLP (Pre)', and each column is a
criterion, either a performance or a fairness metric. Approach 2 is kept commented out below.

Usage:
    python atopsis_analysis.py

-> Paper A-TOPSIS:
https://www.sciencedirect.com/science/article/pii/S187705091501529X?via%3Dihub

-> Paper TOPSIS:
https://link.springer.com/book/10.1007/978-3-642-48318-9

Author: Matheus Becali Rocha
Email: matheusbecali@gmail.com
"""

import pandas as pd

from src.decision_making.a_topsis import ATOPSIS

# from src.decision_making.topsis import TOPSIS

########################################################################################################################
# Approach 1: using the csv files in "../test/avg_mat.csv" and in "../test/std_mat.csv"
########################################################################################################################
# print("-" * 50)
# print("- Approach 1:")
# print("-" * 50)

for _dataset_name in ["db-pad-ufes-20", "db-midas","db-hiba"]:

    # The 'Algorithms' column holds the alternatives, e.g. 'MLP (Pre)'
    df_data = pd.read_csv(f'./results/decision_making/{_dataset_name}_avg_all.csv', delimiter=',')
    alg_names = df_data.Algorithms.values

    # The criteria depend on the dataset: PAD-UFES-20 has two protected
    # attributes (gender and fitzpatrick), the other two only have gender.
    #
    # avg_cost_ben tells A-TOPSIS the direction of each criterion, in the same
    # order as bench_col_names:
    #   'b' = benefit, higher is better -> the performance metrics, and
    #         Disparate Impact, which is best when closest to 1
    #   'c' = cost, lower is better     -> the disparity metrics, which are
    #         stored as absolute values and are best when closest to 0
    if _dataset_name == "db-pad-ufes-20":
        bench_col_names = [
            "Accuracy Score",
            "Balanced Accuracy Score",
            "Precision Score",
            "Recall Score",
            "F1 Score",
            "Statistical Parity (gender)",
            "Disparate Impact (gender)",
            "Equal Opportunity Diff (gender)",
            "Average Odds Diff (gender)",
            "Statistical Parity (fitzpatrick)",
            "Disparate Impact (fitzpatrick)",
            "Equal Opportunity Diff (fitzpatrick)",
            "Average Odds Diff (fitzpatrick)",
        ]
        avg_cost_ben = ['b','b','b','b','b',
                        'c','b','c','c',
                        'c','b','c','c']
    else:
        bench_col_names = [
            "Accuracy Score",
            "Balanced Accuracy Score",
            "Precision Score",
            "Recall Score",
            "F1 Score",
            "Statistical Parity (gender)",
            "Disparate Impact (gender)",
            "Equal Opportunity Diff (gender)",
            "Average Odds Diff (gender)",
        ]
        avg_cost_ben = ['b','b','b','b','b',
                        'c','b','c','c']
        
    save_path_atopsis = f"./plots/ATOPSIS_{_dataset_name}_prob_true_normalize.pdf"

    # std_cost_ben="cost": a lower standard deviation is always better, since it
    # means the result is more stable across the folds.
    # weights=[0.5, 0.5]: the average and the standard deviation matrices weigh
    # the same in the final ranking.
    # normalize=True: puts every criterion on a comparable scale, which matters
    # because the metrics here live in different ranges.
    atop = ATOPSIS(f"./results/decision_making/{_dataset_name}_avg_all.csv",
                   f"./results/decision_making/{_dataset_name}_std_all.csv",
                   alg_col_name="Algorithms", avg_cost_ben=avg_cost_ben,
                   std_cost_ben="cost", weights=[0.5, 0.5],
                   bench_col_names=bench_col_names, normalize=True)

    # Rank the alternatives; verbose=True prints the ranking to stdout
    ranks = atop.get_ranking(verbose=True)

    # show=False only writes the PDF, so the loop never blocks on a window
    atop.plot_ranking(alg_names=alg_names, save_path=save_path_atopsis, show=False)
    print("-" * 50)
    print()

    ########################################################################################################################
    # Approach 2: using TOPSIS
    ########################################################################################################################
    # print("-" * 50)
    # print("- Approach 2:")
    # print("-" * 50)
    # save_path_topsis = f"./plots/TOPSIS_{_dataset_name}.pdf"
    # tp = TOPSIS(f"./results/decision_making/{_dataset_name}_avg_all.csv", avg_cost_ben, 
    #             weights=None, alt_col_name="Algorithms", crit_col_names=bench_col_names)
    # tp.get_closeness_coefficient(verbose=True)
    # tp.plot_ranking(alt_names=alg_names, save_path=save_path_topsis, show=False)
    # print("-" * 50)
    # print("")

    # break
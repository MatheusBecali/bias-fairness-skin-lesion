bash ./run.sh
bash ./run_aggregate_results.sh
bash ./run_latex.sh
python generate_decision_making_csv.py --output-dir ./results/decision_making
python atopsis_analysis.py 
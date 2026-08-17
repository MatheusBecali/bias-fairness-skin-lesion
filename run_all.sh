bash ./run.sh
python aggregate_results.py
bash ./run_latex.sh
python generate_decision_making_csv.py --output-dir ./results/decision_making
python atopsis_analysis.py 
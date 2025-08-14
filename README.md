# Insurance Charges Regression (Keras)

Predict medical insurance charges from tabular features using Keras.
Includes log-target training, Huber loss, decile weighting, and post-hoc linear calibration.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/insurance_regression_keras.py --mode train --csv data/insurance.csv
python src/insurance_regression_keras.py --mode predict --input_csv examples/my_people.csv --out_csv artifacts/my_preds.csv
python scripts/analyze_insurance_results.py --pred auto

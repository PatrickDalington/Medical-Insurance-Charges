# src/insurance_regression_keras.py
# Train / Save / Predict for insurance charges (USD), with post-hoc linear calibration.
# - Saves Keras model to: artifacts/insurance_model.keras
# - Also exports TensorFlow SavedModel dir: artifacts/insurance_model_savedmodel/

import json
import argparse
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import joblib
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber

# ---------------------------
# Project paths (repo-aware)
# ---------------------------
def infer_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / "src").exists() and (cand / "examples").exists():
            return cand
    return Path.cwd()

ROOT = infer_project_root()
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Global config
# ---------------------------
SEED = 35
USE_LOG_TARGET = True      # train on log1p(charges) for long-tail stability
USE_HUBER_LOSS = True
LEARNING_RATE = 1e-3
EPOCHS = 1000
BATCH_SIZE = 32
PATIENCE = 50

MODEL_PATH = ARTIFACTS / "insurance_model.keras"              # Keras 3 native format (file)
SAVEDMODEL_DIR = ARTIFACTS / "insurance_model_savedmodel"     # TF SavedModel (directory)
PREPROCESSOR_PATH = ARTIFACTS / "preprocessor.joblib"
CONFIG_PATH = ARTIFACTS / "config.json"
HISTORY_CSV = ARTIFACTS / "training_history.csv"
PREDICTIONS_CSV = ARTIFACTS / "insurance_predictions.csv"
DEFAULT_TRAIN_CSV = ROOT / "data" / "insurance.csv"
DEFAULT_EXAMPLE_INPUT = ROOT / "examples" / "my_people.csv"
DEFAULT_PRED_OUT = ARTIFACTS / "predictions_out.csv"

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------
# Features
# ---------------------------
BASE_FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
TARGET_COL = "charges"

NUM_COLS = ["age", "bmi", "children", "bmi2", "age2", "bmi_age"]
CAT_COLS = ["sex", "smoker", "region"]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["bmi2"] = d["bmi"] ** 2
    d["age2"] = d["age"] ** 2
    d["bmi_age"] = d["bmi"] * d["age"]
    return d

def build_preprocessor():
    # Handle new/old scikit-learn APIs
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", ohe, CAT_COLS),
        ],
        remainder="drop",
    )

def build_model(n_features: int) -> Sequential:
    m = Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.10),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    loss_fn = Huber(delta=2000.0) if USE_HUBER_LOSS else "mse"
    m.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=loss_fn, metrics=["mae"])
    return m

def apply_calibration(y_usd: np.ndarray, calib: dict | None) -> np.ndarray:
    if not calib:
        return y_usd
    a = float(calib.get("a", 1.0))
    b = float(calib.get("b", 0.0))
    y = a * y_usd + b
    return np.clip(y, 0.0, None)

# ---------------------------
# Train
# ---------------------------
def train(csv_path: Path):
    df = pd.read_csv(csv_path)
    X = engineer_features(df[BASE_FEATURES].copy())
    y_usd = df[TARGET_COL].copy()

    # Split: test (20%), then from remaining make validation (20% of train)
    X_train_full, X_test, y_train_full_usd, y_test_usd = train_test_split(
        X, y_usd, test_size=0.20, random_state=SEED
    )
    X_train, X_val, y_train_usd, y_val_usd = train_test_split(
        X_train_full, y_train_full_usd, test_size=0.20, random_state=SEED
    )

    pre = build_preprocessor()
    X_train_prep = pre.fit_transform(X_train)
    X_val_prep   = pre.transform(X_val)
    X_test_prep  = pre.transform(X_test)

    # Targets (log or usd for training units)
    if USE_LOG_TARGET:
        y_train_fit = np.log1p(y_train_usd.values)
        y_val_fit   = np.log1p(y_val_usd.values)
    else:
        y_train_fit = y_train_usd.values
        y_val_fit   = y_val_usd.values

    model = build_model(X_train_prep.shape[1])

    early = EarlyStopping(monitor="val_loss", mode="min",
                          patience=PATIENCE, restore_best_weights=True, verbose=1)
    reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                               patience=10, min_lr=1e-5, verbose=1)

    # Gentle decile weights with a small top boost
    dec = pd.qcut(y_train_usd.values, 10, labels=False, duplicates="drop")
    w = 1.0 + 0.06 * dec
    w += 0.08 * (dec == 8).astype(float)
    w += 0.12 * (dec == 9).astype(float)
    w = np.clip(w, 1.0, 2.0)

    history = model.fit(
        X_train_prep, y_train_fit,
        validation_data=(X_val_prep, y_val_fit),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early, reduce],
        verbose=1,
        sample_weight=w
    )

    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(HISTORY_CSV, index=False)
    print(f"Saved: {HISTORY_CSV}")

    # ---- Post-hoc linear calibration fitted on VALIDATION predictions ----
    val_fit = model.predict(X_val_prep, verbose=0).flatten()
    y_val_pred_usd_raw = np.expm1(val_fit) if USE_LOG_TARGET else val_fit

    lin = LinearRegression()
    lin.fit(y_val_pred_usd_raw.reshape(-1, 1), y_val_usd.values)
    calib = {"a": float(lin.coef_[0]), "b": float(lin.intercept_)}
    print(f"Calibration (val): a={calib['a']:.6f}, b={calib['b']:.2f}")

    # ---- Evaluate on TEST: raw vs calibrated ----
    test_fit = model.predict(X_test_prep, verbose=0).flatten()
    y_test_pred_usd_raw = np.expm1(test_fit) if USE_LOG_TARGET else test_fit
    y_test_pred_usd_cal = apply_calibration(y_test_pred_usd_raw, calib)

    def report(tag, yhat):
        mae = mean_absolute_error(y_test_usd.values, yhat)
        rmse = sqrt(mean_squared_error(y_test_usd.values, yhat))
        r2 = r2_score(y_test_usd.values, yhat)
        print(f"\n=== Test Metrics ({tag}, USD) ===")
        print(f"MAE:  ${mae:,.2f}\nRMSE: ${rmse:,.2f}\nR^2:  {r2:.4f}")

    report("RAW", y_test_pred_usd_raw)
    report("CALIBRATED", y_test_pred_usd_cal)

    # Save predictions CSV with both columns
    PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    preview = pd.DataFrame({
        "actual_usd": y_test_usd.values,
        "predicted_usd_raw": y_test_pred_usd_raw,
        "predicted_usd_calibrated": y_test_pred_usd_cal,
        "abs_error_usd_calibrated": np.abs(y_test_usd.values - y_test_pred_usd_cal),
    }).reset_index(drop=True)
    preview.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Saved: {PREDICTIONS_CSV}")

    # ---- Save artifacts ----
    assert str(MODEL_PATH).endswith(".keras"), "MODEL_PATH must end with .keras"
    model.save(MODEL_PATH)
    print(f"Saved Keras model → {MODEL_PATH.resolve()}")

    # Export SavedModel
    SAVEDMODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.export(SAVEDMODEL_DIR)
    print(f"Exported SavedModel → {SAVEDMODEL_DIR.resolve()}/")

    # Preprocessor + config
    joblib.dump(pre, PREPROCESSOR_PATH)
    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "use_log_target": USE_LOG_TARGET,
            "base_features": BASE_FEATURES,
            "num_cols": NUM_COLS,
            "cat_cols": CAT_COLS,
            "calibration": calib,
            "apply_calibration_by_default": True
        }, f)
    print(f"Saved preprocessor → {PREPROCESSOR_PATH.resolve()}, config → {CONFIG_PATH.resolve()}")

# ---------------------------
# Load & inference
# ---------------------------
def load_artifacts():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists() or not CONFIG_PATH.exists():
        raise FileNotFoundError("Artifacts not found. Run training first.")
    model = tf.keras.models.load_model(MODEL_PATH)  # .keras file
    pre = joblib.load(PREPROCESSOR_PATH)
    cfg = json.load(open(CONFIG_PATH))
    return model, pre, cfg

def prepare_features_for_inference(df_features: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(df_features)

def predict_csv(input_csv: Path, out_csv: Path, no_calibrate: bool = False):
    model, pre, cfg = load_artifacts()
    use_log = cfg.get("use_log_target", True)
    apply_cal = cfg.get("apply_calibration_by_default", True) and (not no_calibrate)
    calib = cfg.get("calibration", None) if apply_cal else None

    df_in = pd.read_csv(input_csv)
    missing = set(BASE_FEATURES) - set(df_in.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = prepare_features_for_inference(df_in[BASE_FEATURES].copy())
    Xp = pre.transform(X)
    y_fit = model.predict(Xp, verbose=0).flatten()
    y_usd_raw = np.expm1(y_fit) if use_log else y_fit
    y_usd = apply_calibration(y_usd_raw, calib)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out = df_in.copy()
    out["predicted_usd"] = y_usd
    out.to_csv(out_csv, index=False)
    print(f"Wrote predictions to {out_csv}")

def predict_one(age:int, sex:str, bmi:float, children:int, smoker:str, region:str, no_calibrate: bool = False) -> float:
    model, pre, cfg = load_artifacts()
    use_log = cfg.get("use_log_target", True)
    apply_cal = cfg.get("apply_calibration_by_default", True) and (not no_calibrate)
    calib = cfg.get("calibration", None) if apply_cal else None

    one = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi, "children": children,
        "smoker": smoker, "region": region
    }])
    one = prepare_features_for_inference(one)
    Xp = pre.transform(one)
    y_fit = model.predict(Xp, verbose=0).flatten()[0]
    y_usd_raw = np.expm1(y_fit) if use_log else y_fit
    y_usd = apply_calibration(np.array([y_usd_raw]), calib)[0]
    return float(y_usd)

# ---------------------------
# CLI
# ---------------------------
def main():
    p = argparse.ArgumentParser(description="Insurance charges model: train / predict (with calibration).")
    p.add_argument("--mode", choices=["train", "predict", "predict_one"], default="train")
    p.add_argument("--csv", default=str(DEFAULT_TRAIN_CSV), help="[train] training CSV path")
    p.add_argument("--input_csv", default=str(DEFAULT_EXAMPLE_INPUT), help="[predict] input CSV with base features")
    p.add_argument("--out_csv", default=str(DEFAULT_PRED_OUT), help="[predict] output CSV path")
    p.add_argument("--no_calibrate", action="store_true", help="Disable calibration at inference")

    # predict_one args
    p.add_argument("--age", type=int)
    p.add_argument("--sex", type=str, choices=["male","female"])
    p.add_argument("--bmi", type=float)
    p.add_argument("--children", type=int)
    p.add_argument("--smoker", type=str, choices=["yes","no"])
    p.add_argument("--region", type=str, choices=["northeast","northwest","southeast","southwest"])

    args = p.parse_args()

    if args.mode == "train":
        train(Path(args.csv))

    elif args.mode == "predict":
        predict_csv(Path(args.input_csv), Path(args.out_csv), no_calibrate=args.no_calibrate)

    elif args.mode == "predict_one":
        needed = [args.age, args.sex, args.bmi, args.children, args.smoker, args.region]
        if any(v is None for v in needed):
            raise SystemExit("--mode predict_one requires --age --sex --bmi --children --smoker --region")
        pred = predict_one(args.age, args.sex, args.bmi, args.children, args.smoker, args.region,
                           no_calibrate=args.no_calibrate)
        print(f"Predicted charges (USD): ${pred:,.2f}")

if __name__ == "__main__":
    main()

# scripts/analyze_insurance_results.py
import argparse
import pandas as pd
import numpy as np
from math import sqrt
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

import matplotlib.pyplot as plt


# ---------------------------
# Repo-aware helpers
# ---------------------------
def infer_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, *here.parents]:
        if (cand / "src").exists() and (cand / "examples").exists():
            return cand
    return Path.cwd()


# ---------------------------
# Metrics helpers
# ---------------------------
def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    abs_err = np.abs(y_true - y_pred)
    p90 = float(np.percentile(abs_err, 90))
    p95 = float(np.percentile(abs_err, 95))
    return {
        "MAE_$": mae,
        "RMSE_$": rmse,
        "R2": r2,
        "Median_AE_$": medae,
        "P90_AE_$": p90,
        "P95_AE_$": p95,
    }


def print_metrics(title, m):
    print(title)
    print(f"MAE:  ${m['MAE_$']:,.2f}")
    print(f"RMSE: ${m['RMSE_$']:,.2f}")
    print(f"R^2:  {m['R2']:.4f}")
    print(f"Median AE: ${m['Median_AE_$']:,.2f}")
    print(f"P90 AE:    ${m['P90_AE_$']:,.2f}")
    print(f"P95 AE:    ${m['P95_AE_$']:,.2f}\n")


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze Keras insurance regression predictions and training history."
    )
    parser.add_argument(
        "--pred",
        choices=["auto", "calibrated", "plain", "raw"],
        default="auto",
        help=(
            "Which prediction column to use: "
            "'auto' (prefer calibrated), 'calibrated' (predicted_usd_calibrated), "
            "'plain' (predicted_usd), 'raw' (predicted_usd_raw)."
        ),
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default=None,
        help="Artifacts directory (default: <repo>/artifacts)",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default=None,
        help="Assets directory for figures (default: <repo>/assets)",
    )
    args = parser.parse_args()

    ROOT = infer_project_root()
    ARTIFACTS = Path(args.artifacts) if args.artifacts else ROOT / "artifacts"
    ASSETS = Path(args.assets) if args.assets else ROOT / "assets"
    ASSETS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    PRED_PATH = ARTIFACTS / "insurance_predictions.csv"
    HIST_PATH = ARTIFACTS / "training_history.csv"

    # ---- Load predictions ----
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Missing {PRED_PATH}. Run training first.")

    pred = pd.read_csv(PRED_PATH)

    # Decide which prediction column to use
    choice_map = {
        "calibrated": ["predicted_usd_calibrated"],
        "plain": ["predicted_usd"],
        "raw": ["predicted_usd_raw"],
        "auto": ["predicted_usd_calibrated", "predicted_usd", "predicted_usd_raw"],
    }
    candidates = choice_map[args.pred]
    pred_col = next((c for c in candidates if c in pred.columns), None)
    if pred_col is None:
        raise ValueError(
            f"{PRED_PATH} does not contain any of the required columns for --pred {args.pred}: {candidates}"
        )

    required_cols = {"actual_usd", pred_col}
    if not required_cols.issubset(pred.columns):
        raise ValueError(f"{PRED_PATH} missing required columns {required_cols}.")

    # Clean
    pred = pred.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required_cols))
    y_true = pred["actual_usd"].to_numpy(dtype=float)
    y_hat = pred[pred_col].to_numpy(dtype=float)
    print(f"Using prediction column: {pred_col}")

    # ---- Sanity warning ----
    if np.median(y_true) > 1000 and np.median(y_hat) < 100:  # rough heuristic
        print("⚠️  WARNING: Predictions look tiny vs actuals. "
              "If you trained on log(charges), inverse-transform before saving.\n")

    # ---- Overall metrics ----
    overall = metrics(y_true, y_hat)
    print_metrics("=== Overall Test Metrics ===", overall)

    # ---- Worst cases ----
    pred["error"] = y_true - y_hat
    pred["abs_error"] = np.abs(pred["error"])
    print("Top 10 worst absolute errors:")
    cols_to_show = ["actual_usd", pred_col, "abs_error"]
    print(pred.sort_values("abs_error", ascending=False).head(10)[cols_to_show])
    print()

    # ---- Tail analysis (top 10% by actuals) ----
    tail_threshold = np.quantile(y_true, 0.90)
    tail_mask = y_true >= tail_threshold
    if tail_mask.any():
        tail = metrics(y_true[tail_mask], y_hat[tail_mask])
        print_metrics(
            f"=== Tail Metrics (top 10% actuals: ≥ ${tail_threshold:,.0f}) ===", tail
        )

    # ---- Decile calibration ----
    deciles = pd.qcut(y_true, q=10, labels=False, duplicates="drop")
    calib = pd.DataFrame(
        {"decile": deciles, "actual": y_true, "pred": y_hat, "abs_err": np.abs(y_true - y_hat)}
    )
    calib_summary = (
        calib.groupby("decile")
        .agg(
            mean_actual_usd=("actual", "mean"),
            mean_pred_usd=("pred", "mean"),
            mae_usd=("abs_err", "mean"),
            count=("abs_err", "size"),
        )
        .reset_index()
    )

    calib_csv = ARTIFACTS / "calibration_by_decile.csv"
    calib_summary.to_csv(calib_csv, index=False)
    print(f"Saved decile calibration table → {calib_csv.relative_to(ROOT)}")

    # ---- Plots → assets/ ----
    # 1) Predicted vs Actual
    plt.figure()
    plt.scatter(y_true, y_hat, s=10)
    mn = float(min(y_true.min(), y_hat.min()))
    mx = float(max(y_true.max(), y_hat.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Actual charges (USD)")
    plt.ylabel("Predicted charges (USD)")
    plt.title("Keras Model: Predicted vs Actual")
    plt.tight_layout()
    out1 = ASSETS / "keras_pred_vs_actual.png"
    plt.savefig(out1)
    plt.close()

    # 2) Residual histogram
    residuals = y_true - y_hat
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (Actual - Predicted) USD")
    plt.ylabel("Count")
    plt.title("Keras Model: Residuals Histogram")
    plt.tight_layout()
    out2 = ASSETS / "keras_residuals_hist.png"
    plt.savefig(out2)
    plt.close()

    # 3) Absolute error vs Actual
    plt.figure()
    plt.scatter(y_true, np.abs(residuals), s=10)
    plt.xlabel("Actual charges (USD)")
    plt.ylabel("Absolute error (USD)")
    plt.title("Absolute Error vs Actual")
    plt.tight_layout()
    out3 = ASSETS / "abs_error_vs_actual.png"
    plt.savefig(out3)
    plt.close()

    # 4) Decile calibration plot
    plt.figure()
    x = calib_summary["decile"].to_numpy()
    plt.plot(x, calib_summary["mean_actual_usd"].to_numpy(), marker="o")
    plt.plot(x, calib_summary["mean_pred_usd"].to_numpy(), marker="o")
    plt.xlabel("Actual charge decile (0 = cheapest … 9 = most expensive)")
    plt.ylabel("Mean USD")
    plt.title("Calibration by Decile: Mean Actual vs Mean Predicted")
    plt.legend(["Mean Actual", "Mean Predicted"])
    plt.tight_layout()
    out4 = ASSETS / "calibration_by_decile.png"
    plt.savefig(out4)
    plt.close()

    # ---- Learning curves (if history present) ----
    if HIST_PATH.exists():
        hist = pd.read_csv(HIST_PATH)
        monitor_col = "val_loss" if "val_loss" in hist.columns else "loss"
        best_idx = int(hist[monitor_col].idxmin())
        best_epoch = best_idx + 1
        best_val = float(hist[monitor_col].min())
        print(f"Best epoch by {monitor_col}: {best_epoch} (value={best_val:,.6f})")

        if {"loss", "val_loss"}.issubset(hist.columns):
            plt.figure()
            plt.plot(hist["loss"])
            plt.plot(hist["val_loss"])
            plt.xlabel("Epoch")
            plt.ylabel("MSE loss")
            plt.title("Learning Curve: Loss")
            plt.legend(["train", "val"])
            plt.tight_layout()
            out5 = ASSETS / "keras_learning_curve_loss.png"
            plt.savefig(out5)
            plt.close()

        if {"mae", "val_mae"}.issubset(hist.columns):
            plt.figure()
            plt.plot(hist["mae"])
            plt.plot(hist["val_mae"])
            plt.xlabel("Epoch")
            plt.ylabel("MAE (USD or log-units)")
            plt.title("Learning Curve: MAE")
            plt.legend(["train", "val"])
            plt.tight_layout()
            out6 = ASSETS / "keras_learning_curve_mae.png"
            plt.savefig(out6)
            plt.close()
    else:
        print(f"Note: {HIST_PATH.relative_to(ROOT)} not found. Re-run training with history saving enabled.")

    # ---- Text summary → artifacts/ ----
    summary_path = ARTIFACTS / "metrics_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Overall Test Metrics ===\n")
        for k, v in overall.items():
            f.write(f"{k}: {v}\n")
        if tail_mask.any():
            f.write("\n=== Tail Metrics (top 10% actuals) ===\n")
            for k, v in tail.items():
                f.write(f"{k}: {v}\n")

    print("\nWrote:")
    for p in [out1, out2, out3, out4, summary_path, calib_csv]:
        try:
            print(" -", p.relative_to(ROOT))
        except Exception:
            print(" -", p)


if __name__ == "__main__":
    main()

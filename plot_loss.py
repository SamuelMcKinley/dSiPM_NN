#!/usr/bin/env python3
"""
Visualize training results from loss_history.csv and validation predictions,
supporting both per-epoch CSVs (val_predictions_epoch_###.csv)
and combined val_predictions_all_epochs.csv.
"""

import os
import re
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def read_csv_as_dicts(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_float(val):
    try:
        return float(val)
    except Exception:
        return np.nan


def plot_with_dashed_lines(ax, x, y, xlabel, ylabel, title):
    ax.plot(x, y, 'b-', lw=2)
    for n in range(50, int(max(x)) + 50, 50):
        ax.axvline(n, color='r', ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, ls=':', alpha=0.6)


def linear_regression(x, y):
    x, y = np.asarray(x), np.asarray(y)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    r_num = np.sum((x - x.mean()) * (y - y.mean()))
    r_den = np.sqrt(np.sum((x - x.mean())**2) * np.sum((y - y.mean())**2))
    r2 = (r_num / r_den)**2 if r_den != 0 else 0.0
    return slope, intercept, r2


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Plot training results without pandas.")
    ap.add_argument("folder", help="Folder containing loss_history.csv and prediction CSV(s)")
    ap.add_argument("--outdir", default="plots", help="Output directory for plots")
    args = ap.parse_args()

    folder = Path(args.folder)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # ------------------ LOSS HISTORY ------------------
    loss_csv = folder / "loss_history.csv"
    if not loss_csv.exists():
        raise FileNotFoundError(f"{loss_csv} not found")

    rows = read_csv_as_dicts(loss_csv)
    epochs = np.array([safe_float(r["epoch"]) for r in rows])
    val_loss = np.array([safe_float(r["val_loss"]) for r in rows])
    train_loss = np.array([safe_float(r["train_loss"]) for r in rows])
    mae = np.array([safe_float(r["val_mae"]) for r in rows])
    rmse = np.array([safe_float(r["val_rmse"]) for r in rows])

    best_idx = np.nanargmin(val_loss)
    best_epoch = int(epochs[best_idx])
    best_loss = val_loss[best_idx]

    # Validation loss vs epoch
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_with_dashed_lines(ax, epochs, val_loss, "Epoch", "Validation Loss (MSE)", "Validation Loss vs Epoch")
    ax.plot(epochs, train_loss, 'g--', lw=1.5, label="Train Loss")
    ax.axvline(best_epoch, color='k', lw=1.2, ls=':', label=f"Best ({best_epoch})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outdir / "validation_loss_vs_epoch.png", dpi=200)
    plt.close(fig)

    # MSE vs epoch
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_with_dashed_lines(ax, epochs, val_loss, "Epoch", "MSE", "MSE vs Epoch")
    plt.tight_layout()
    fig.savefig(outdir / "mse_vs_epoch.png", dpi=200)
    plt.close(fig)

    # MAE/RMSE vs epoch
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, mae, 'b-', lw=2, label="MAE")
    ax.plot(epochs, rmse, color='orange', lw=2, label="RMSE")
    for n in range(50, int(max(epochs)) + 50, 50):
        ax.axvline(n, color='r', ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (GeV)")
    ax.set_title("Validation Errors vs Epoch")
    ax.legend()
    ax.grid(True, ls=':')
    plt.tight_layout()
    fig.savefig(outdir / "mae_rmse_vs_epoch.png", dpi=200)
    plt.close(fig)

    # ------------------ PREDICTION SCATTER ------------------
    val_files = sorted(glob.glob(str(folder / "val_predictions_epoch_*.csv")))
    combined_file = folder / "val_predictions_all_epochs.csv"

    if val_files:
        # Use last epoch (highest number)
        epoch_nums = [int(re.search(r"epoch_(\d+)", f).group(1)) for f in val_files]
        latest_idx = np.argmax(epoch_nums)
        val_csv = val_files[latest_idx]
        print(f"Using predictions from {val_csv}")
        with open(val_csv, "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    elif combined_file.exists():
        print(f"Using predictions from {combined_file}")
        with open(combined_file, "r") as f:
            reader = csv.DictReader(f)
            all_data = list(reader)

        # Handle combined file (multiple epochs)
        # Use the last epoch available
        epoch_col = "epoch" if "epoch" in all_data[0] else None
        if epoch_col:
            epochs_in_file = [int(r["epoch"]) for r in all_data if r.get("epoch", "").isdigit()]
            last_epoch = max(epochs_in_file)
            data = [r for r in all_data if int(r["epoch"]) == last_epoch]
            print(f"Extracted last epoch = {last_epoch} ({len(data)} samples)")
        else:
            data = all_data  # fallback: use all if no epoch column
    else:
        raise FileNotFoundError(f"No validation prediction CSVs found in {folder}")

    y_true = np.array([safe_float(r["true_energy"]) for r in data])
    y_pred = np.array([safe_float(r["pred_energy"]) for r in data])
    dev = y_pred - y_true

    # Linear regression (manual)
    slope, intercept, r2 = linear_regression(y_true, y_pred)
    y_fit = slope * y_true + intercept

    # Predicted vs True plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=20, alpha=0.7, label="Samples")
    ax.plot(y_true, y_true, 'k--', lw=1.5, label="Ideal (y=x)")
    ax.plot(y_true, y_fit, 'r-', lw=2,
            label=f"Fit: y={slope:.3f}x+{intercept:.3f}\nR²={r2:.4f}")
    ax.set_xlabel("True Energy (GeV)")
    ax.set_ylabel("Predicted Energy (GeV)")
    ax.set_title("Predicted vs True Energy")
    ax.legend()
    ax.grid(True, ls=':')
    plt.tight_layout()
    fig.savefig(outdir / "pred_vs_true.png", dpi=250)
    plt.close(fig)

    # Residuals plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_true, dev, s=20, alpha=0.7)
    ax.axhline(0, color='k', ls='--')
    ax.set_xlabel("True Energy (GeV)")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title("Residuals vs True Energy")
    ax.grid(True, ls=':')
    plt.tight_layout()
    fig.savefig(outdir / "residuals_vs_true.png", dpi=250)
    plt.close(fig)

    print(f"✅ Plots saved to {outdir.resolve()}")


if __name__ == "__main__":
    main()
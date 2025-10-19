#!/usr/bin/env python3
"""
Plots loss curves and energy predictions from consolidated CSVs
(no pandas version).
"""

import os, csv, re, glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def read_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_float(v):
    try:
        return float(v)
    except:
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
    r2 = (r_num / r_den)**2 if r_den != 0 else 0
    return slope, intercept, r2


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Plot results from single CSV output.")
    ap.add_argument("folder", help="Folder containing loss_history.csv and val_predictions_all_epochs.csv")
    ap.add_argument("--outdir", default="plots", help="Output folder for plots")
    args = ap.parse_args()

    folder = Path(args.folder)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # --- Loss history ---
    loss_csv = folder / "loss_history.csv"
    if not loss_csv.exists():
        raise FileNotFoundError(f"{loss_csv} not found")

    rows = read_csv(loss_csv)
    epochs = np.array([safe_float(r["epoch"]) for r in rows])
    val_loss = np.array([safe_float(r["val_loss"]) for r in rows])
    train_loss = np.array([safe_float(r["train_loss"]) for r in rows])
    mae = np.array([safe_float(r["val_mae"]) for r in rows])
    rmse = np.array([safe_float(r["val_rmse"]) for r in rows])

    best_idx = np.nanargmin(val_loss)
    best_epoch = int(epochs[best_idx])

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

    # MAE/RMSE
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, mae, 'b-', lw=2, label="MAE")
    ax.plot(epochs, rmse, 'orange', lw=2, label="RMSE")
    for n in range(50, int(max(epochs)) + 50, 50):
        ax.axvline(n, color='r', ls='--', lw=0.8, alpha=0.6)
    ax.legend()
    ax.grid(True, ls=':')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (GeV)")
    ax.set_title("Validation Errors vs Epoch")
    plt.tight_layout()
    fig.savefig(outdir / "mae_rmse_vs_epoch.png", dpi=200)
    plt.close(fig)

    # --- Predictions ---
    val_csv = folder / "val_predictions_all_epochs.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"{val_csv} not found")

    rows = read_csv(val_csv)
    epochs_all = np.array([safe_float(r["epoch"]) for r in rows])
    y_true = np.array([safe_float(r["true_energy"]) for r in rows])
    y_pred = np.array([safe_float(r["pred_energy"]) for r in rows])

    # Select latest epoch for scatter plot
    last_epoch = int(np.nanmax(epochs_all))
    mask = epochs_all == last_epoch
    y_true_last = y_true[mask]
    y_pred_last = y_pred[mask]

    slope, intercept, r2 = linear_regression(y_true_last, y_pred_last)
    y_fit = slope * y_true_last + intercept

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_last, y_pred_last, s=20, alpha=0.7, label=f"Epoch {last_epoch}")
    ax.plot(y_true_last, y_true_last, 'k--', lw=1.5, label="Ideal y=x")
    ax.plot(y_true_last, y_fit, 'r-', lw=2, label=f"Fit: y={slope:.3f}x+{intercept:.3f}\nR²={r2:.4f}")
    ax.set_xlabel("True Energy (GeV)")
    ax.set_ylabel("Predicted Energy (GeV)")
    ax.set_title("Predicted vs True Energy")
    ax.legend()
    ax.grid(True, ls=':')
    plt.tight_layout()
    fig.savefig(outdir / "pred_vs_true.png", dpi=250)
    plt.close(fig)

    # Residuals plot (latest epoch)
    dev = y_pred_last - y_true_last
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_true_last, dev, s=20, alpha=0.7)
    ax.axhline(0, color='k', ls='--')
    ax.set_xlabel("True Energy (GeV)")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title(f"Residuals vs True Energy (Epoch {last_epoch})")
    ax.grid(True, ls=':')
    plt.tight_layout()
    fig.savefig(outdir / "residuals_vs_true.png", dpi=250)
    plt.close(fig)

    print(f"✅ Plots saved to {outdir.resolve()}")

if __name__ == "__main__":
    main()
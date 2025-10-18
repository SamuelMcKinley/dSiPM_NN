#!/usr/bin/env python3
"""
Visualize training results from loss_history.csv and val_predictions_epoch_###.csv.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def plot_with_dashed_lines(ax, x, y, xlabel, ylabel, title):
    ax.plot(x, y, 'b-', lw=2)
    for n in range(50, int(max(x)) + 50, 50):
        ax.axvline(n, color='r', ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, ls=':', alpha=0.6)


def main():
    parser_description = "Plot loss curves and energy predictions from NN training output."
    import argparse
    ap = argparse.ArgumentParser(description=parser_description)
    ap.add_argument("folder", help="Folder containing loss_history.csv and val_predictions_epoch_###.csv")
    ap.add_argument("--outdir", default="plots", help="Directory to save plots")
    args = ap.parse_args()

    folder = Path(args.folder)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # ------------------- LOSS HISTORY -------------------
    loss_csv = folder / "loss_history.csv"
    if not loss_csv.exists():
        raise FileNotFoundError(f"{loss_csv} not found")

    df = pd.read_csv(loss_csv)
    epochs = df["epoch"]
    val_loss = df["val_loss"]
    train_loss = df["train_loss"]
    mae = df["val_mae"]
    rmse = df["val_rmse"]

    best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
    best_loss = df["val_loss"].min()

    # 1. Validation loss vs epoch
    fig, ax = plt.subplots(figsize=(8,5))
    plot_with_dashed_lines(ax, epochs, val_loss, "Epoch", "Validation Loss (MSE)", "Validation Loss vs Epoch")
    ax.plot(epochs, train_loss, 'g--', lw=1.5, label="Train Loss")
    ax.legend()
    ax.axvline(best_epoch, color='k', lw=1.5, ls=':', label=f"Best ({best_epoch})")
    plt.tight_layout()
    fig.savefig(outdir / "validation_loss_vs_epoch.png", dpi=200)
    plt.close(fig)

    # 2. MSE (same as val_loss) vs epoch (redundant but separate figure)
    fig, ax = plt.subplots(figsize=(8,5))
    plot_with_dashed_lines(ax, epochs, val_loss, "Epoch", "MSE", "MSE vs Epoch")
    plt.tight_layout()
    fig.savefig(outdir / "mse_vs_epoch.png", dpi=200)
    plt.close(fig)

    # 3. MAE / RMSE curves
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(epochs, mae, 'b-', lw=2, label="MAE")
    ax.plot(epochs, rmse, 'orange', lw=2, label="RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (GeV)")
    ax.set_title("Validation Errors vs Epoch")
    for n in range(50, int(max(epochs))+50, 50):
        ax.axvline(n, color='r', ls='--', lw=0.8, alpha=0.6)
    ax.legend()
    ax.grid(True, ls=':')
    plt.tight_layout()
    fig.savefig(outdir / "mae_rmse_vs_epoch.png", dpi=200)
    plt.close(fig)

    # ------------------- PREDICTION SCATTER -------------------
    val_files = sorted(glob.glob(str(folder / "val_predictions_epoch_*.csv")))
    if not val_files:
        raise FileNotFoundError(f"No val_predictions_epoch_###.csv files found in {folder}")

    # Use the latest (or best) one
    epoch_nums = [int(re.search(r"epoch_(\d+)", f).group(1)) for f in val_files]
    latest_idx = np.argmax(epoch_nums)
    val_csv = val_files[latest_idx]
    print(f"Using predictions from {val_csv}")

    pred_df = pd.read_csv(val_csv)
    y_true = pred_df["true_energy"].values
    y_pred = pred_df["pred_energy"].values
    dev = y_pred - y_true

    # Fit linear regression (Pred vs True)
    reg = LinearRegression().fit(y_true.reshape(-1,1), y_pred)
    y_fit = reg.predict(y_true.reshape(-1,1))
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_true, y_pred, s=20, alpha=0.7, label="Samples")
    ax.plot(y_true, y_true, 'k--', lw=1.5, label="Ideal (y=x)")
    ax.plot(y_true, y_fit, 'r-', lw=2, label=f"Fit: y={slope:.3f}x+{intercept:.3f}\nR²={r2:.4f}")
    ax.set_xlabel("True Energy (GeV)")
    ax.set_ylabel("Predicted Energy (GeV)")
    ax.set_title("Predicted vs True Energy")
    ax.grid(True, ls=':')
    ax.legend()
    plt.tight_layout()
    fig.savefig(outdir / "pred_vs_true.png", dpi=250)
    plt.close(fig)

    # 4. Residuals plot
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(y_true, dev, s=20, alpha=0.7)
    ax.axhline(0, color='k', ls='--')
    ax.set_xlabel("True Energy (GeV)")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title("Residuals vs True Energy")
    ax.grid(True, ls=':')
    plt.tight_layout()
    fig.savefig(outdir / "residuals_vs_true.png", dpi=250)
    plt.close(fig)

    print(f"✅ Plots saved in {outdir.resolve()}")

if __name__ == "__main__":
    main()
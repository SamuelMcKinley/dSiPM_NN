#!/usr/bin/env python3
import os, csv, argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="Directory containing loss_history.csv (e.g. NN_photons_model_70x70)")
    args = ap.parse_args()

    loss_csv = os.path.join(args.model_dir, "loss_history.csv")
    if not os.path.exists(loss_csv):
        raise SystemExit(f"Missing {loss_csv}")

    epochs, tr, va = [], [], []
    with open(loss_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                epochs.append(int(row["epoch"]))
                tr.append(float(row["train_loss"]))
                va.append(float(row["val_loss"]))
            except Exception:
                continue

    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.plot(epochs, tr, label="train_loss")
    plt.plot(epochs, va, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Train/Val Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/loss_vs_epoch.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
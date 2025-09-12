#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

# Use a non-interactive backend (safer on HPC without DISPLAY)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot loss history from training log.")
    p.add_argument("loss_file", help="Path to loss_history_<energy>.txt")
    p.add_argument("--interval", type=int, default=50, help="Plot every N epochs incrementally")
    p.add_argument("--outdir", type=str, default="loss_plots", help="Directory to save loss plots")
    return p.parse_args()


def read_loss_file(path: Path):
    epochs, train, val = [], [], []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue  # skip malformed lines
            try:
                e = int(parts[0])
                tr = float(parts[1])
                va = float(parts[2])
            except ValueError:
                continue  # skip non-numeric lines (e.g., headers)
            epochs.append(e)
            train.append(tr)
            val.append(va)

    # Sort by epoch just in case
    order = np.argsort(epochs)
    epochs = list(np.array(epochs)[order])
    train = list(np.array(train)[order])
    val = list(np.array(val)[order])
    return epochs, train, val


def compute_increments(total_epochs: int, interval: int):
    """Generate plot endpoints: interval, 2*interval, ..., up to total_epochs"""
    increments = []
    current = interval
    
    # Add multiples of interval
    while current < total_epochs:
        increments.append(current)
        current += interval
    
    # Always add the final epoch count
    increments.append(total_epochs)
    
    return sorted(set(increments))  # Remove duplicates and sort


def plot_incremental(epochs, train_losses, val_losses, energy, outdir: Path, interval: int):
    outdir.mkdir(parents=True, exist_ok=True)
    total_epochs = len(epochs)

    print(f"[plot_loss] total_epochs={total_epochs}, interval={interval}")
    increments = compute_increments(total_epochs, interval)
    print(f"[plot_loss] increments to generate: {increments}")

    for end_epoch in increments:
        # Get data up to end_epoch
        sub_train = train_losses[:end_epoch]
        sub_val = val_losses[:end_epoch]
        x = np.arange(1, end_epoch + 1)  # epoch numbers 1 to end_epoch

        plt.figure(figsize=(12, 6))
        
        # Plot as lines (more typical for loss curves)
        plt.plot(x, sub_train, 'b-', linewidth=2, label="Train Loss", alpha=0.8)
        plt.plot(x, sub_val, 'r-', linewidth=2, label="Validation Loss", alpha=0.8)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss History (Epochs 1-{end_epoch}) | Energy = {energy} GeV")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        plt.xlim(1, end_epoch)
        
        # Optional: set y-axis to start from 0 or auto-scale nicely
        y_min = min(min(sub_train), min(sub_val))
        y_max = max(max(sub_train), max(sub_val))
        plt.ylim(max(0, y_min * 0.95), y_max * 1.05)

        filename = outdir / f"loss_{int(energy)}GeV_epochs_1-{end_epoch}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[plot_loss] ✅ Saved {filename}")


def main():
    args = parse_args()
    loss_path = Path(args.loss_file).resolve()
    outdir = Path(args.outdir).resolve()
    print(f"[plot_loss] Using loss file: {loss_path}")
    print(f"[plot_loss] Output directory: {outdir}")

    # Parse energy from filename like ".../loss_history_5.txt"
    energy_str = loss_path.stem.split("_")[-1]
    try:
        energy = float(energy_str)
    except ValueError:
        print(f"[plot_loss] ⚠️ Could not extract energy from filename: {loss_path.name}")
        return

    epochs, train_losses, val_losses = read_loss_file(loss_path)
    if not epochs:
        print("[plot_loss] ⚠️ No epochs parsed from file (empty or malformed).")
        return

    print(f"[plot_loss] Loaded {len(epochs)} epochs of data")
    plot_incremental(epochs, train_losses, val_losses, energy, outdir, args.interval)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Analyze NN loss history and predictions.")
    p.add_argument("spad_size", help="SPAD size label (e.g. '4000x4000')")
    p.add_argument("folder", help="Folder containing loss_history_*.txt files")
    return p.parse_args()

def extract_energy_from_filename(fname):
    # Assumes filenames like loss_history_10.txt or loss_history_mixed.txt
    m = re.search(r"loss_history_(\d+)", fname)
    if m:
        return float(m.group(1))
    else:
        # Fallback for mixed or non-numeric runs
        print(f"Non-numeric energy found in {fname}, treating as 'mixed'")
        return -1.0  # sentinel value for mixed files

def main():
    args = parse_args()
    spad = args.spad_size
    folder = args.folder

    energies = []
    preds_mean = []
    preds_std = []
    actuals = []

    # find all txt files
    files = [f for f in os.listdir(folder) if f.startswith("loss_history_") and f.endswith(".txt")]
    if not files:
        raise FileNotFoundError(f"No loss_history_*.txt files found in {folder}")

    outdir = os.path.join(folder, f"analysis_{spad}")
    os.makedirs(outdir, exist_ok=True)

    for f in sorted(files, key=lambda x: extract_energy_from_filename(x)):
        energy = extract_energy_from_filename(f)
        path = os.path.join(folder, f)

        # load columns: epoch,train,val,pred,true
        data = np.loadtxt(path, delimiter=",", skiprows=0)
        if data.ndim == 1:  # only one row
            data = data.reshape(1, -1)

        E_pred = data[:, 3]
        E_actual = data[:, 4]

        mean_pred = np.mean(E_pred)
        std_pred = np.std(E_pred)

        energies.append(energy)
        preds_mean.append(mean_pred)
        preds_std.append(std_pred)
        actuals.append(np.mean(E_actual))  # all rows should have same actual, but take mean for safety

        # Histogram plot per energy
        energy_label = f"{energy:.0f} GeV" if energy >= 0 else "Mixed"

        plt.figure()
        plt.hist(E_pred, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
        plt.axvline(E_actual[0], color="red", linestyle="--", label=f"True {E_actual[0]:.2f}")
        plt.xlabel("Predicted Energy (GeV)")
        plt.ylabel("Counts")
        plt.title(f"SPAD {spad} | Energy {energy_label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_{spad}_{energy_label.replace(' ', '')}.png"))
        plt.close()

    # E_pred vs E_actual plot
    plt.figure()
    plt.errorbar(actuals, preds_mean, yerr=preds_std, fmt="o", capsize=5, label="Predicted")
    min_e = min(min(actuals), min(preds_mean))
    max_e = max(max(actuals), max(preds_mean))
    plt.plot([min_e, max_e], [min_e, max_e], "r--", label="Ideal y=x")
    plt.xlabel("True Energy (GeV)")
    plt.ylabel("Mean Predicted Energy (GeV)")
    plt.title(f"SPAD {spad} | Prediction vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"pred_vs_actual_{spad}.png"))
    plt.close()

    print(f"Analysis complete. Results saved in {outdir}")

if __name__ == "__main__":
    main()

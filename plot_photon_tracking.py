import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    # --- Parse command line argument ---
    if len(sys.argv) < 2:
        print("❌ Usage: python plot_photon_tracking.py <input_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = "photon_tracking_plots"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_csv):
        print(f"❌ Error: {input_csv} not found.")
        sys.exit(1)

    # --- Data storage ---
    data = defaultdict(lambda: {"energy": [], "total": [], "lost": []})

    # --- Read CSV ---
    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spad = row["SPAD_Size"]
            energy = float(row["Energy"])
            total = int(row["Total_Photons"])
            lost = int(row["Lost_Photons"])

            data[spad]["energy"].append(energy)
            data[spad]["total"].append(total)
            data[spad]["lost"].append(lost)

    # --- Sort by energy for each SPAD ---
    for spad in data.keys():
        order = np.argsort(data[spad]["energy"])
        for key in ["energy", "total", "lost"]:
            data[spad][key] = np.array(data[spad][key])[order]

    # --- Plot Detected Photons vs Energy ---
    for spad, vals in data.items():
        energy = vals["energy"]
        total = vals["total"]
        lost = vals["lost"]
        detected = total - lost

        plt.figure(figsize=(7, 5))
        plt.plot(energy, detected, "o-", lw=2, label="Detected Photons")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.title(f"Detected Photons vs Energy ({spad})")
        plt.xlabel("Beam Energy (GeV)")
        plt.ylabel("Detected Photons (Total - Lost)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"detected_vs_energy_{spad}.png"))
        plt.close()

    # --- Plot Lost Photons vs Energy ---
    for spad, vals in data.items():
        energy = vals["energy"]
        lost = vals["lost"]

        plt.figure(figsize=(7, 5))
        plt.plot(energy, lost, "s--", lw=2, color="orange", label="Lost Photons")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.title(f"Lost Photons vs Energy ({spad})")
        plt.xlabel("Beam Energy (GeV)")
        plt.ylabel("Lost Photons")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"lost_vs_energy_{spad}.png"))
        plt.close()

    # --- Combined comparison across SPADs (Detected) ---
    plt.figure(figsize=(8, 6))
    for spad, vals in data.items():
        energy = vals["energy"]
        detected = np.array(vals["total"]) - np.array(vals["lost"])
        plt.plot(energy, detected, marker="o", lw=2, label=f"{spad} Detected")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("Detected Photons vs Energy (All SPADs)")
    plt.xlabel("Beam Energy (GeV)")
    plt.ylabel("Detected Photons")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_detected_all_SPADs.png"))
    plt.close()

    # --- Combined comparison across SPADs (Lost) ---
    plt.figure(figsize=(8, 6))
    for spad, vals in data.items():
        energy = vals["energy"]
        plt.plot(energy, vals["lost"], marker="s", lw=2, label=f"{spad} Lost")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("Lost Photons vs Energy (All SPADs)")
    plt.xlabel("Beam Energy (GeV)")
    plt.ylabel("Lost Photons")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_lost_all_SPADs.png"))
    plt.close()

    print(f"✅ Plots saved to: {output_dir}/")

if __name__ == "__main__":
    main()
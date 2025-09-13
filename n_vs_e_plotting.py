#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # for HPC/non-GUI environments
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Plot nPhotons vs Energy from CSV.")
    parser.add_argument("csv_file", help="CSV file with Energy_GeV and nPhotons columns")
    parser.add_argument("spad_size", help="SPAD size string (e.g., 2000x2000)")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv_file)

    if "Energy_GeV" not in df.columns or "nPhotons" not in df.columns:
        raise ValueError("CSV must contain columns 'Energy_GeV' and 'nPhotons'")

    # Sort by energy to get a clean line
    df = df.sort_values("Energy_GeV")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df["Energy_GeV"], df["nPhotons"], marker="o", linestyle="-", label=f"SPAD {args.spad_size}")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("nPhotons")
    plt.title(f"nPhotons vs Energy ({args.spad_size})")
    plt.grid(True)
    plt.legend()

    # Save plot
    out_file = Path(f"nPhotons_vs_Energy_{args.spad_size}.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"✅ Saved plot as {out_file}")

if __name__ == "__main__":
    main()


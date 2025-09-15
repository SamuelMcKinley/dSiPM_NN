#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # safe for HPC
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Plot nPhotons vs Energy for each time slice.")
    parser.add_argument("csv_file", help="CSV file with Energy_GeV, TimeSlice, nPhotons")
    parser.add_argument("spad_size", help="SPAD size string (e.g., 2000x2000)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    required = {"Energy_GeV", "TimeSlice", "nPhotons"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    # For each time slice, make a plot
    for t in sorted(df["TimeSlice"].unique()):
        df_t = df[df["TimeSlice"] == t].sort_values("Energy_GeV")

        plt.figure(figsize=(8, 6))
        plt.plot(df_t["Energy_GeV"], df_t["nPhotons"], marker="o", linestyle="-",
                 label=f"Time slice {t}")
        plt.xlabel("Energy (GeV)")
        plt.ylabel("nPhotons")
        plt.title(f"nPhotons vs Energy (Time slice {t}, SPAD {args.spad_size})")
        plt.grid(True)
        plt.legend()

        out_file = Path(f"nPhotons_vs_Energy_slice{t}_{args.spad_size}.png")
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"✅ Saved {out_file}")

if __name__ == "__main__":
    main()


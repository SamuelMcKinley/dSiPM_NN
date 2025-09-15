#!/usr/bin/env python3
import argparse
import csv
import matplotlib
matplotlib.use("Agg")   # safe for HPC
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Plot nPhotons vs Energy for each time slice.")
    parser.add_argument("csv_file", help="CSV file with Energy_GeV,TimeSlice,nPhotons")
    parser.add_argument("spad_size", help="SPAD size string (e.g., 2000x2000)")
    args = parser.parse_args()

    # Read CSV manually
    time_data = defaultdict(list)  # maps timeslice -> list of (energy, nPhotons)
    with open(args.csv_file, newline="") as f:
        reader = csv.DictReader(f)
        required = {"Energy_GeV", "TimeSlice", "nPhotons"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {required}")
        for row in reader:
            energy = float(row["Energy_GeV"])
            t = int(row["TimeSlice"])
            nphot = int(row["nPhotons"])
            time_data[t].append((energy, nphot))

    # For each time slice, plot energy vs photons
    for t, entries in sorted(time_data.items()):
        # Sort by energy for a clean line
        entries.sort(key=lambda x: x[0])
        energies = [e for e, _ in entries]
        nphotons = [n for _, n in entries]

        plt.figure(figsize=(8, 6))
        plt.plot(energies, nphotons, marker="o", linestyle="-",
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


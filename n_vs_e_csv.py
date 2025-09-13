#!/usr/bin/env python3
import argparse
import numpy as np
import csv
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Sum photons in tensor and append to CSV.")
    parser.add_argument("tensor_file", help="Path to the .npy tensor file")
    parser.add_argument("energy", type=float, help="Energy value (GeV)")
    parser.add_argument("--csv", default="photon_counts.csv",
                        help="CSV file to append results (default: photon_counts.csv)")
    args = parser.parse_args()

    # Load tensor
    tensor = np.load(args.tensor_file)
    nPhotons = int(np.sum(tensor))

    print(f"Loaded tensor {args.tensor_file} with shape {tensor.shape}")
    print(f"Total photons: {nPhotons} at energy {args.energy} GeV")

    # Append to CSV
    csv_path = Path(args.csv)
    new_file = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        # Write header if new file
        if new_file:
            writer.writerow(["Energy_GeV", "nPhotons"])
        writer.writerow([args.energy, nPhotons])

    print(f"✅ Appended to {args.csv}")

if __name__ == "__main__":
    main()



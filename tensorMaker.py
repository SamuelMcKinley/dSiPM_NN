import os
import sys
import re
import numpy as np
import ROOT
import csv
import fcntl
from datetime import datetime

# Deadtime Boolean
Deadtime = True

# ROOT setup
ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)

# Script & repo paths
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))  # <— repo path for consistent outputs

# Adjustable geometry settings
xShift = np.array([3.7308, 3.5768, 3.8008, 3.6468])
yShift = np.array([-3.7293, -3.6878, -3.6878, -3.6488])
shrink_rules = [(0.1 + 0.4 * i, round(0.23 * i, 2)) for i in range(40)]
limits = np.array([rule[0] for rule in shrink_rules])
shift_amounts = np.array([rule[1] for rule in shrink_rules])
DET_MIN = -100.0
DET_MAX =  100.0
DET_WIDTH = DET_MAX - DET_MIN

def shrink_toward_center_array(vals: np.ndarray) -> np.ndarray:
    abs_vals = np.abs(vals)
    idx = np.searchsorted(limits, abs_vals, side="right")
    idx = np.clip(idx, 0, len(shift_amounts) - 1)
    return vals - shift_amounts[idx] * np.sign(vals)

# Classes
class Photons:
    def __init__(self, event):
        self.time_final = np.array(event.OP_time_final)
        self.array4Sorting = np.argsort(self.time_final)
        self.time_final = self.time_final[self.array4Sorting]
        self.productionFiber = self._arr(event.OP_productionFiber)
        self.isCoreC = self._arr(event.OP_isCoreC)
        self.pos_final_x = self._arr(event.OP_pos_final_x)
        self.pos_final_y = self._arr(event.OP_pos_final_y)
        self.pos_final_z = self._arr(event.OP_pos_final_z)
        self.pos_produced_z = self._arr(event.OP_pos_produced_z)
        self.w = np.ones(self.nPhotons(), dtype=np.float32)

    def _arr(self, var): return np.array(var)[self.array4Sorting]
    def nPhotons(self): return len(self.pos_final_x)

class SiPMInfo:
    def __init__(self, channelSize, nBins):
        self.channelSize = channelSize
        self.nBins = nBins
        self.name = f"{int(channelSize)}x{int(channelSize)}"

def getNBins(l, h, s): return int((h - l) / s)

# CSV Updating
def update_photon_tracking(spad_size, energy, total_photons, lost_photons):
    """
    Append to a single canonical CSV in the repo directory (SCRIPT_DIR),
    regardless of where the script is launched from.
    Each run adds its own line; no overwriting.
    """
    csv_path = os.path.join(SCRIPT_DIR, "photon_tracking.csv")
    header = ["SPAD_Size", "Energy", "Total_Photons", "Lost_Photons"]

    # Create new file if missing, then append
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow([
            spad_size,
            f"{float(energy):g}",
            str(int(total_photons)),
            str(int(lost_photons))
        ])

    print(f"Appended entry to {csv_path} for {spad_size}, {energy} GeV")


def append_photon_energy_row(spad_size: str, nPhotons: int, energy: float):
    """
    Append (nPhotons, Energy) to a SPAD-specific CSV in SCRIPT_DIR.
    Safe for parallel runs using an OS-level file lock.
    """
    csv_path = os.path.join(SCRIPT_DIR, f"photon_energy_{spad_size}.csv")
    header = ["nPhotons", "Energy"]

    # Open in append+read mode so we can check size under lock
    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            is_empty = (f.tell() == 0)

            writer = csv.writer(f)
            if is_empty:
                writer.writerow(header)

            writer.writerow([int(nPhotons), float(energy)])

            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main():
    if len(sys.argv) < 5:
        print("Usage: python tensorMaker.py <root_file> <energy> <output_folder> <SPAD_size>")
        sys.exit(1)

    input_file_path, energy, output_folder, spad_size = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4].strip()

    # Validate SPAD size strictly
    if not re.match(r"^\d+x\d+$", spad_size):
        print(f"Invalid SPAD size '{spad_size}'. Expected like '2000x2000'.")
        sys.exit(1)

    try:
        side_length = int(spad_size.split("x")[0])
        spacing = side_length / 1000.0
    except Exception:
        print(f"Invalid SPAD size format '{spad_size}'. Use '70x70' style.")
        sys.exit(1)

    sipm = SiPMInfo(side_length, getNBins(DET_MIN, DET_MAX, spacing))
    input_file = ROOT.TFile(input_file_path, "READ")
    tree = input_file.Get("tree")

    os.makedirs(os.path.join(output_folder, "npy"), exist_ok=True)
    csv_path = os.path.join(output_folder, "labels.csv")
    write_header = not os.path.exists(csv_path)

    total_photons_cumulative, total_lost_cumulative, nEvents = 0, 0, -1

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["filename", "energy"])

        # Semi-arbitrary slices chosen to best visualize detector map
        time_slice_ranges = [(0, 9), (9, 9.5), (9.5, 10), (10, 15), (15, 40)]

        for event in tree:
            nEvents += 1
            g = Photons(event)
            print(f"Event {nEvents}: {g.nPhotons()} raw photons")

            # Apply spatial corrections and detector acceptance mask
            x_raw = g.pos_final_x + np.take(xShift, g.productionFiber)
            y_raw = g.pos_final_y + np.take(yShift, g.productionFiber)
            x_shifted = shrink_toward_center_array(x_raw)
            y_shifted = shrink_toward_center_array(y_raw)

            mask = (g.isCoreC.astype(bool)) & (g.pos_final_z > 0) & (0.0 < g.time_final) & (g.time_final < 40.0)
            x_vals = 10 * x_shifted[mask]
            y_vals = 10 * y_shifted[mask]
            t_vals = g.time_final[mask]
            w_vals = g.w[mask]

            # Count only photons that reach detector plane before deadtime
            total_photons_cumulative += len(x_vals)

            photons_lost = 0
            if Deadtime:
                ix = ((x_vals - DET_MIN) / (DET_WIDTH / sipm.nBins)).astype(int)
                iy = ((y_vals - DET_MIN) / (DET_WIDTH / sipm.nBins)).astype(int)
                ix = np.clip(ix, 0, sipm.nBins - 1)
                iy = np.clip(iy, 0, sipm.nBins - 1)

                # Vectorized
                pixel_ids = iy * sipm.nBins + ix
                _, first_indices = np.unique(pixel_ids, return_index=True)
                accepted = np.zeros_like(t_vals, dtype=bool)
                accepted[first_indices] = True

                photons_after = int(np.count_nonzero(accepted))
                photons_lost = int(len(t_vals) - photons_after)
                total_lost_cumulative += photons_lost

                print(f"Event {nEvents}: raw={g.nPhotons()}, reach={len(x_vals)}, kept={photons_after}, lost={photons_lost}")
                x_vals, y_vals, t_vals, w_vals = (
                    x_vals[accepted], y_vals[accepted], t_vals[accepted], w_vals[accepted]
                )

                nPhotons_used = photons_after  # <-- what the detector "keeps" under deadtime
            else:
                print(f"Event {nEvents}: {len(t_vals)} photons used (Deadtime OFF)")
                nPhotons_used = int(len(t_vals))

            # Append one training row (safe under parallel runs)
            append_photon_energy_row(spad_size, nPhotons_used, energy)

            # Build 3D histogram tensor
            hist_tensor = []
            for t_low, t_high in time_slice_ranges:
                mask_t = (t_vals >= t_low) & (t_vals < t_high)
                H, _ = np.histogramdd(
                    np.stack((y_vals[mask_t], x_vals[mask_t]), axis=-1),
                    bins=(sipm.nBins, sipm.nBins),
                    range=[[DET_MIN, DET_MAX], [DET_MIN, DET_MAX]],
                    weights=w_vals[mask_t]
                )
                hist_tensor.append(H.astype(np.float32))

            event_tensor = np.stack(hist_tensor, axis=0)
            filename = f"event_{nEvents:04d}_ch{sipm.name}.npy"
            np.save(os.path.join(output_folder, "npy", filename), event_tensor)
            writer.writerow([filename, energy])

    # Append CSV entry
    update_photon_tracking(spad_size, energy, total_photons_cumulative, total_lost_cumulative)

    # Meta logging
    meta_path = os.path.join(SCRIPT_DIR, f"tensor_meta_{spad_size}_{energy:.1f}GeV.txt")
    with open(meta_path, "w") as m:
        m.write(f"Run timestamp: {datetime.now()}\n")
        m.write(f"SPAD_Size: {spad_size}\nEnergy: {energy} GeV\n")
        m.write(f"Events processed: {nEvents + 1}\n")
        m.write(f"Total photons reaching detector: {total_photons_cumulative}\n")
        m.write(f"Total photons lost (deadtime): {total_lost_cumulative}\n")
    print(f"Wrote metadata log of most recent event to {meta_path}")

if __name__ == "__main__":
    main()
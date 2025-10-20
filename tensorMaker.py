import os
import sys
import numpy as np
import ROOT
import csv
import fcntl  # for safe file locking
from datetime import datetime

# ---------------- Deadtime Boolean ----------------
Deadtime = True

# ---------------- ROOT setup ----------------
ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)

# ---------------- Geometry settings ----------------
xShift = np.array([3.7308, 3.5768, 3.8008, 3.6468])
yShift = np.array([-3.7293, -3.6878, -3.6878, -3.6488])
shrink_rules = [(0.1 + 0.4 * i, round(0.23 * i, 2)) for i in range(40)]
limits = np.array([rule[0] for rule in shrink_rules])
shift_amounts = np.array([rule[1] for rule in shrink_rules])

def shrink_toward_center_array(vals: np.ndarray) -> np.ndarray:
    abs_vals = np.abs(vals)
    idx = np.searchsorted(limits, abs_vals, side="right")
    idx = np.clip(idx, 0, len(shift_amounts) - 1)
    return vals - shift_amounts[idx] * np.sign(vals)

# ---------------- Classes ----------------
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

# ---------------- CSV Updating ----------------
def update_photon_tracking(spad_size, energy, total_photons, lost_photons):
    """Append or update photon_tracking.csv cumulatively with locking."""
    csv_path = "photon_tracking.csv"

    # Create new file if missing
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["SPAD_Size", "Energy", "Total_Photons", "Lost_Photons"])
        print("📁 Created new photon_tracking.csv")

    with open(csv_path, "r+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        reader = csv.reader(f)
        data = list(reader)

        # Validate header
        if not data or data[0] != ["SPAD_Size", "Energy", "Total_Photons", "Lost_Photons"]:
            data = [["SPAD_Size", "Energy", "Total_Photons", "Lost_Photons"]]

        matching_rows = [r for r in data[1:] if len(r) >= 2 and r[0] == spad_size and r[1] == str(energy)]
        print(f"🔍 Before update: found {len(matching_rows)} existing rows for {spad_size}, {energy} GeV")

        updated = False
        for row in data[1:]:
            if len(row) < 4: 
                continue
            try:
                if row[0] == spad_size and float(row[1]) == float(energy):
                    old_total, old_lost = int(row[2]), int(row[3])
                    row[2] = str(old_total + int(total_photons))
                    row[3] = str(old_lost + int(lost_photons))
                    print(f"✏️  Updated existing entry: total {old_total}→{row[2]}, lost {old_lost}→{row[3]}")
                    updated = True
                    break
            except ValueError:
                continue

        if not updated:
            data.append([spad_size, str(energy), str(int(total_photons)), str(int(lost_photons))])
            print(f"➕ Added new entry for {spad_size}, {energy} GeV")

        f.seek(0); f.truncate()
        csv.writer(f).writerows(data)
        fcntl.flock(f, fcntl.LOCK_UN)

    print(f"📊 photon_tracking.csv updated for {spad_size}, {energy} GeV")

# ---------------- Main ----------------
def main():
    if len(sys.argv) < 5:
        print("❌ Usage: python tensorMaker.py <root_file> <energy> <output_folder> <SPAD_size>")
        sys.exit(1)

    input_file_path, energy, output_folder, spad_size = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]

    try:
        side_length = int(spad_size.split("x")[0])
        spacing = side_length / 1000.0
    except Exception:
        print(f"❌ Invalid SPAD size format '{spad_size}'. Use '70x70' style.")
        sys.exit(1)

    sipm = SiPMInfo(side_length, getNBins(-80.0, 80.0, spacing))
    input_file = ROOT.TFile(input_file_path, "READ")
    tree = input_file.Get("tree")

    os.makedirs(os.path.join(output_folder, "npy"), exist_ok=True)
    csv_path = os.path.join(output_folder, "labels.csv")
    write_header = not os.path.exists(csv_path)

    total_photons_cumulative, total_lost_cumulative, nEvents = 0, 0, -1

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header: writer.writerow(["filename", "energy"])
        time_slice_ranges = [(0, 9), (9, 9.5), (9.5, 10), (10, 15), (15, 40)]

        for event in tree:
            nEvents += 1
            g = Photons(event)
            total_photons_cumulative += g.nPhotons()
            print(f"📦 Event {nEvents}: {g.nPhotons()} raw photons")

            # Apply spatial corrections
            x_raw = g.pos_final_x + np.take(xShift, g.productionFiber)
            y_raw = g.pos_final_y + np.take(yShift, g.productionFiber)
            x_shifted, y_shifted = shrink_toward_center_array(x_raw), shrink_toward_center_array(y_raw)
            mask = (g.isCoreC.astype(bool)) & (g.pos_final_z > 0) & (0.0 < g.time_final) & (g.time_final < 40.0)
            x_vals, y_vals, t_vals, w_vals = 10*x_shifted[mask], 10*y_shifted[mask], g.time_final[mask], g.w[mask]

            photons_lost = 0
            if Deadtime:
                ix = ((x_vals + 80.0) / (160.0 / sipm.nBins)).astype(int)
                iy = ((y_vals + 80.0) / (160.0 / sipm.nBins)).astype(int)
                ix, iy = np.clip(ix, 0, sipm.nBins - 1), np.clip(iy, 0, sipm.nBins - 1)

                active = np.ones((sipm.nBins, sipm.nBins), dtype=bool)
                accepted = np.zeros_like(t_vals, dtype=bool)
                for i in range(len(t_vals)):
                    x_i, y_i = ix[i], iy[i]
                    if active[y_i, x_i]:
                        accepted[i] = True
                        active[y_i, x_i] = False
                photons_after = np.count_nonzero(accepted)
                photons_lost = len(t_vals) - photons_after
                total_lost_cumulative += photons_lost
                print(f"Event {nEvents}: {photons_after} kept, {photons_lost} lost (Deadtime ON)")
                x_vals, y_vals, t_vals, w_vals = x_vals[accepted], y_vals[accepted], t_vals[accepted], w_vals[accepted]
            else:
                print(f"Event {nEvents}: {len(t_vals)} photons used (Deadtime OFF)")

            # Build 3D histogram tensor
            hist_tensor = []
            for t_low, t_high in time_slice_ranges:
                mask_t = (t_vals >= t_low) & (t_vals < t_high)
                H, _ = np.histogramdd(np.stack((y_vals[mask_t], x_vals[mask_t]), axis=-1),
                                      bins=(sipm.nBins, sipm.nBins),
                                      range=[[-80.0, 80.0], [-80.0, 80.0]],
                                      weights=w_vals[mask_t])
                hist_tensor.append(H.astype(np.float32))
            event_tensor = np.stack(hist_tensor, axis=0)
            filename = f"event_{nEvents:04d}_ch{sipm.name}.npy"
            np.save(os.path.join(output_folder, "npy", filename), event_tensor)
            writer.writerow([filename, energy])

    update_photon_tracking(spad_size, energy, total_photons_cumulative, total_lost_cumulative)

    # --- Meta logging ---
    meta_path = f"tensor_meta_{spad_size}_{energy:.1f}GeV.txt"
    with open(meta_path, "w") as m:
        m.write(f"Run timestamp: {datetime.now()}\n")
        m.write(f"SPAD_Size: {spad_size}\nEnergy: {energy} GeV\n")
        m.write(f"Events processed: {nEvents + 1}\n")
        m.write(f"Total photons (raw): {total_photons_cumulative}\n")
        m.write(f"Total photons lost: {total_lost_cumulative}\n")
    print(f"🧾 Wrote metadata to {meta_path}")

if __name__ == "__main__":
    main()
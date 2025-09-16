import os
import sys
import numpy as np
import ROOT
import csv

# ROOT settings
ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)

# Position shifts per fiber (to center fiber groups)
xShift = [3.7308, 3.5768, 3.8008, 3.6468]
yShift = [-3.7293, -3.6878, -3.6878, -3.6488]

# (distance_limit_from_center, shift_amount_toward_center)
shrink_rules = [(0.1 + 0.4 * i, round(0.23 * i, 2)) for i in range(40)]

def shrink_toward_center(val: float) -> float:
    abs_val = abs(val)
    for limit, shift in shrink_rules:
        if abs_val <= limit:
            return val - shift * np.sign(val)
    return val - 2.0 * np.sign(val)

class Photons:
    def __init__(self, event):
        self.time_final = np.array(event.OP_time_final)
        self.array4Sorting = np.argsort(self.time_final)
        self.time_final = self.time_final[self.array4Sorting]
        self.productionFiber = self.getArrayFromEvent(event.OP_productionFiber)
        self.isCoreC = self.getArrayFromEvent(event.OP_isCoreC)
        self.pos_final_x = self.getArrayFromEvent(event.OP_pos_final_x)
        self.pos_final_y = self.getArrayFromEvent(event.OP_pos_final_y)
        self.pos_final_z = self.getArrayFromEvent(event.OP_pos_final_z)
        self.pos_produced_z = self.getArrayFromEvent(event.OP_pos_produced_z)
        self.w = np.full(self.nPhotons(), 1.0)

    def getArrayFromEvent(self, var):
        return np.array(var)[self.array4Sorting]

    def nPhotons(self):
        return len(self.pos_final_x)

    def x(self, i):
        raw_x = self.pos_final_x[i] + xShift[self.productionFiber[i]]
        return shrink_toward_center(raw_x)

    def y(self, i):
        raw_y = self.pos_final_y[i] + yShift[self.productionFiber[i]]
        return shrink_toward_center(raw_y)

    def z(self, i):
        return self.pos_produced_z[i]

    def zEnd(self, i):
        return self.pos_final_z[i]

    def t(self, i):
        return self.time_final[i]

    def fiberNumber(self, i):
        return self.productionFiber[i]

class SiPMInfo:
    def __init__(self, channelSize, nBins):
        self.channelSize = channelSize  # microns
        self.nBins = nBins
        self.name = f"{int(channelSize)}x{int(channelSize)}"

def getNBins(l, h, s):
    return int((h - l) / s)

def main():
    if len(sys.argv) < 5:
        print("\u274c Error: Usage: python torchMaker.py <root_file> <energy> <output_folder> <SPAD_size>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    energy = float(sys.argv[2])
    output_folder = sys.argv[3]
    spad_size = sys.argv[4]

    # Derive spacing from SPAD size
    try:
        side_length = int(spad_size.split("x")[0])
        spacing = side_length / 1000.0  # convert microns to mm
    except Exception as e:
        print(f"\u274c Invalid SPAD size format '{spad_size}'. Use something like '70x70'.")
        sys.exit(1)

    sipm = SiPMInfo(side_length, getNBins(-80.0, 80.0, spacing))

    input_file = ROOT.TFile(input_file_path, "READ")
    tree = input_file.Get("tree")

    os.makedirs(os.path.join(output_folder, "npy"), exist_ok=True)
    csv_path = os.path.join(output_folder, "labels.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["filename", "energy"])

        time_slice_ranges = [(0, 9), (9, 9.5), (9.5, 10), (10, 15), (15, 40)]

        nEvents = -1
        for event in tree:
            nEvents += 1
            g = Photons(event)
            if nEvents % 5 == 0:
                print(f"Event {nEvents}: {g.nPhotons()} photons")

            # --- Photon selection (vectorized) ---
            x_raw = g.pos_final_x + np.take(xShift, g.productionFiber)
            y_raw = g.pos_final_y + np.take(yShift, g.productionFiber)

            x_shifted = np.vectorize(shrink_toward_center)(x_raw)
            y_shifted = np.vectorize(shrink_toward_center)(y_raw)

            z_end = g.pos_final_z
            t_all = g.time_final
            is_core = g.isCoreC.astype(bool)

            mask = (is_core) & (z_end > 0) & (0.0 < t_all) & (t_all < 40.0)

            x_vals = 10 * x_shifted[mask]
            y_vals = 10 * y_shifted[mask]
            t_vals = t_all[mask]
            w_vals = g.w[mask]


            hist_tensor = []
            for t_low, t_high in time_slice_ranges:
                mask = (t_vals >= t_low) & (t_vals < t_high)
                H, _ = np.histogramdd(
                    sample=np.stack((y_vals[mask], x_vals[mask]), axis=-1),
                    bins=(sipm.nBins, sipm.nBins),
                    range=[[-80.0, 80.0], [-80.0, 80.0]],
                    weights=w_vals[mask]
                )
                hist_tensor.append(H.astype(np.float32))

            event_tensor = np.stack(hist_tensor, axis=0)
            filename = f"event_{nEvents:04d}_ch{sipm.name}.npy"
            out_path = os.path.join(output_folder, "npy", filename)
            np.save(out_path, event_tensor)
            writer.writerow([filename, energy])

if __name__ == '__main__':
    main()

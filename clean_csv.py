import csv
from collections import defaultdict

totals = defaultdict(lambda: [0, 0])  # (Total_Photons, Lost_Photons)

with open("photon_tracking.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Skip repeated headers
        if row["Total_Photons"] == "Total_Photons":
            continue
        key = (row["SPAD_Size"], row["Energy"])
        totals[key][0] += int(row["Total_Photons"])
        totals[key][1] += int(row["Lost_Photons"])

with open("photon_tracking_combined.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["SPAD_Size", "Energy", "Total_Photons", "Lost_Photons"])
    for (spad, energy), (tot, lost) in sorted(totals.items()):
        writer.writerow([spad, energy, tot, lost])

print("Combined CSV written to photon_tracking_combined.csv")
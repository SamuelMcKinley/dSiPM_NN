import csv
import numpy as np
import torch
from torch.utils.data import Dataset

class PhotonCountEnergyDataset(Dataset):
    """
    CSV columns expected: nPhotons, Energy
    """
    def __init__(self, csv_path: str, transform: str = "log1p"):
        self.csv_path = csv_path
        self.transform = transform

        nphot, en = [], []
        with open(csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    n = float(row["nPhotons"])
                    e = float(row["Energy"])
                except Exception:
                    continue
                nphot.append(n)
                en.append(e)

        if len(nphot) == 0:
            raise RuntimeError(f"No valid rows found in {csv_path}")

        self.nphot = np.asarray(nphot, dtype=np.float32)
        self.energy = np.asarray(en, dtype=np.float32)

        # Feature transform (still scalar)
        if self.transform == "log1p":
            self.x_raw = np.log1p(self.nphot)
        elif self.transform == "none":
            self.x_raw = self.nphot.copy()
        else:
            raise ValueError("transform must be 'log1p' or 'none'")

    def __len__(self):
        return len(self.energy)

    def get_all_energies(self):
        return self.energy.tolist()

    def get_all_features(self):
        return self.x_raw.copy()

    def __getitem__(self, idx):
        x = torch.tensor([self.x_raw[idx]], dtype=torch.float32)   # shape (1,)
        y = torch.tensor(self.energy[idx], dtype=torch.float32)    # scalar
        return x, y
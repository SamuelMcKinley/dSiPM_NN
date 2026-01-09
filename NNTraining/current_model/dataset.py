#!/usr/bin/env python3
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

FNAME_RE = re.compile(
    r"""^tensor_            # prefix
        (?P<idx>[^_]+)_     # event index or id
        (?P<particle>[^_]+)_
        (?P<energy>[\d.]+)_ # energy in GeV (float ok)
        (?P<group>[^_]+)_
        (?P<spad>[^_]+)     # SPAD label, e.g., 4000x4000
        \.npz$""",
    re.VERBOSE,
)


def _parse_energy_from_name(p: Path) -> float:
    m = FNAME_RE.match(p.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {p.name}")
    return float(m.group("energy"))


def _to_chw(arr: np.ndarray) -> np.ndarray:
    """
    Ensure output is (C, H, W).
    Accepts (H, W) -> (1, H, W)
            (C, H, W) -> unchanged
            (H, W, C) -> transpose to (C, H, W) when C is small-ish
    """
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        if arr.shape[-1] <= 16 and arr.shape[0] != arr.shape[-1]:
            return np.transpose(arr, (2, 0, 1))
        return arr
    raise ValueError(f"Unsupported tensor shape {arr.shape}; expected 2D or 3D")


class PhotonEnergyDataset(Dataset):
    """
    Loads tensors and lnN from:
      tensor_{i}_{particle}_{energy}_{group}_{spad}.npz

    In .npz:
      - x   : normalized tensor (C,H,W) float32 (already divided by denom)
      - lnN : float32 (ln of total kept photons)
    """

    def __init__(self, tensor_path: str, dtype: np.dtype = np.float32):
        path = Path(tensor_path)

        if path.is_file() and path.suffix.lower() == ".npz":
            self.files = [path]
        elif path.is_dir():
            # only .npz
            self.files = [p for p in path.iterdir() if p.suffix.lower() == ".npz"]
        else:
            raise FileNotFoundError(f"tensor_path not found or not a .npz: {tensor_path}")

        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {tensor_path}")

        self.dtype = dtype

        # Peek first file to infer channels/shape (C,H,W)
        p0 = self.files[0]
        with np.load(p0, allow_pickle=False) as z:
            arr0 = np.asarray(z["x"], dtype=self.dtype)
        arr0 = _to_chw(arr0)
        self.channels, self.height, self.width = arr0.shape

        # Pre-extract energies and names
        self._energies: List[float] = []
        self._names: List[str] = []
        for p in self.files:
            self._energies.append(_parse_energy_from_name(p))
            self._names.append(p.name)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        p = self.files[idx]
        e = self._energies[idx]
        name = self._names[idx]

        with np.load(p, allow_pickle=False) as z:
            arr = np.asarray(z["x"], dtype=self.dtype)
            lnN = np.asarray(z["lnN"], dtype=np.float32).reshape(1)  # (1,)

        arr = _to_chw(arr)
        x = torch.from_numpy(arr)                 # (C,H,W) float32
        lnN_t = torch.from_numpy(lnN)             # (1,) float32
        y = torch.tensor(e, dtype=torch.float32)  # scalar (linear energy in GeV)

        return x, lnN_t, y, name

    def get_all_energies(self) -> List[float]:
        return list(self._energies)

    def get_all_names(self) -> List[str]:
        return list(self._names)
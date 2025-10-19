#!/usr/bin/env python3
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

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
        \.npy$""",
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
        # If last dim looks like channels, move it to first
        if arr.shape[-1] <= 16 and arr.shape[0] != arr.shape[-1]:
            return np.transpose(arr, (2, 0, 1))
        # Already (C,H,W)
        return arr
    raise ValueError(f"Unsupported tensor shape {arr.shape}; expected 2D or 3D")


class PhotonEnergyDataset(Dataset):
    """
    Loads tensors from a folder (or a single .npy file) and predicts linear energy.
    Energies are extracted from filenames:
      tensor_{i}_{particle}_{energy}_{group}_{spad}.npy
    """

    def __init__(
        self,
        tensor_path: str,
        mmap: bool = True,
        dtype: np.dtype = np.float32,
    ):
        path = Path(tensor_path)
        if path.is_file() and path.suffix.lower() == ".npy":
            self.files = [path]
        elif path.is_dir():
            # DO NOT sort (preserve "random" input order as requested)
            self.files = [p for p in path.iterdir() if p.suffix.lower() == ".npy"]
        else:
            raise FileNotFoundError(f"tensor_path not found or not a .npy: {tensor_path}")

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {tensor_path}")

        self.mmap = mmap
        self.dtype = dtype

        # Peek first file to infer shape/channels
        arr0 = np.load(self.files[0], mmap_mode="r" if mmap else None)
        arr0 = _to_chw(np.asarray(arr0, dtype=self.dtype))
        self.channels, self.height, self.width = arr0.shape

        # Pre-extract energies and keep names
        self._energies: List[float] = []
        self._names: List[str] = []
        for p in self.files:
            e = _parse_energy_from_name(p)
            self._energies.append(e)
            self._names.append(p.name)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        p = self.files[idx]
        e = self._energies[idx]
        name = self._names[idx]

        arr = np.load(p, mmap_mode="r" if self.mmap else None).astype(self.dtype, copy=False)
        arr = _to_chw(arr)  # (C,H,W)
        x = torch.from_numpy(arr)  # float32
        y = torch.tensor(e, dtype=torch.float32)  # linear energy target

        return x, y, name

    def get_all_energies(self) -> List[float]:
        return list(self._energies)

    def get_all_names(self) -> List[str]:
        return list(self._names)

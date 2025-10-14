import os
from pathlib import Path
from typing import List, Tuple, Union
import re
import numpy as np
import torch
from torch.utils.data import Dataset


class PhotonEnergyDataset(Dataset):
    """
    Loads photon tensors from a .npy file or a directory of .npy files.

    Expected array shapes per sample:
      - (T, H, W)  -> T treated as channels (C=T)
      - (H, W)     -> promoted to (1, H, W)

    Returns (x, y) where:
      x: float32 tensor [C, H, W]
      y: float32 scalar tensor (energy in chosen target space)

    Args:
      tensor_path: path to .npy or directory of .npy files
      energy: float (GeV) or None (will parse from filename if None)
      target_space: "linear" or "log"
      mmap: use numpy memmap for lower RAM footprint
    """

    def __init__(
        self,
        tensor_path: Union[str, os.PathLike],
        energy: Union[float, None] = None,  # allow None for mixed
        target_space: str = "log",
        mmap: bool = True,
    ):
        self.root = Path(tensor_path)
        if self.root.is_dir():
            self.files: List[Path] = sorted(p for p in self.root.glob("*.npy") if p.is_file())
        else:
            if self.root.suffix.lower() != ".npy":
                raise ValueError(f"Expected .npy or directory, got: {self.root}")
            self.files = [self.root]

        if not self.files:
            raise FileNotFoundError(f"No .npy files found under {self.root}")

        # Extract energies from filenames if energy is None
        self.energies: List[float] = []
        if energy is None:
            pattern = re.compile(
                r"tensor_\d+_[A-Za-z0-9\+\-]+_([\d\.]+)_\d+_[A-Za-z0-9x]+\.npy"
            )
            for f in self.files:
                m = pattern.match(f.name)
                if not m:
                    raise ValueError(
                        f"Filename {f.name} does not match expected pattern "
                        "(tensor_i_particle_energy_group_SPAD.npy)"
                    )
                self.energies.append(float(m.group(1)))
        else:
            self.energies = [float(energy)] * len(self.files)

        # Save linear and log targets
        self.energy_linear = np.array(self.energies, dtype=np.float32)

        if target_space not in ("linear", "log"):
            raise ValueError("target_space must be 'linear' or 'log'")
        self.target_space = target_space
        self.mmap = mmap

        # Inspect first sample to record dims
        arr0 = self._load_array(self.files[0])
        x0 = self._to_CHW(arr0)
        self.channels, self.height, self.width = x0.shape

    def __len__(self) -> int:
        return len(self.files)

    def _load_array(self, path: Path) -> np.ndarray:
        return np.load(path, mmap_mode="r" if self.mmap else None)

    def _to_CHW(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = arr[None, :, :]  # (H, W) -> (1, H, W)
        elif arr.ndim == 3:
            pass  # (T, H, W)
        else:
            raise ValueError(f"Unsupported tensor shape {arr.shape}, expected (H,W) or (T,H,W)")
        return arr

    def _target_value(self, idx: int) -> float:
        E = self.energy_linear[idx]
        if self.target_space == "linear":
            return float(E)
        else:
            eps = 1e-8  # avoid log(0)
            return float(np.log(E + eps))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        arr = self._load_array(path)
        chw = self._to_CHW(np.asarray(arr, dtype=np.float32, order="C"))

        x = torch.from_numpy(chw)  # [C, H, W]
        y = torch.tensor(self._target_value(idx), dtype=torch.float32)  # scalar
        return x, y

    # Helper to expose all energies (for train.py logging)
    def get_all_energies(self):
        return self.energy_linear.tolist()
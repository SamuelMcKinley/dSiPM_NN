import os
from pathlib import Path
from typing import List, Tuple, Union
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
      energy: float (GeV)
      target_space: "linear" or "log"
      mmap: use numpy memmap for lower RAM footprint
    """
    def __init__(
        self,
        tensor_path: Union[str, os.PathLike],
        energy: float,
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

        self.energy_linear = float(energy)
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
        if self.mmap:
            arr = np.load(path, mmap_mode="r")
        else:
            arr = np.load(path)
        return arr

    def _to_CHW(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            # (H, W) -> (1, H, W)
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            # assume (T, H, W) -> (C, H, W)
            pass
        else:
            raise ValueError(f"Unsupported tensor shape {arr.shape}, expected (H,W) or (T,H,W)")
        return arr

    def _target_value(self) -> float:
        if self.target_space == "linear":
            return self.energy_linear
        else:
            eps = 1e-8  # avoid log(0)
            return float(np.log(self.energy_linear + eps))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        arr = self._load_array(path)
        chw = self._to_CHW(np.asarray(arr, dtype=np.float32, order="C"))

        x = torch.from_numpy(chw)  # [C, H, W]
        y = torch.tensor(self._target_value(), dtype=torch.float32)  # scalar
        return x, y


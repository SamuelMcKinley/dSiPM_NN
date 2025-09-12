#!/usr/bin/env python3
"""
Simple prediction script. Works with either:
  - TorchScript:  <outdir>/predictor.ts
  - State dict:   <outdir>/best.pth  (uses your local model.py)

Usage:
  python3 predict.py /path/to/sample.npy NN_model_4000x4000
  python3 predict.py /path/to/dir_of_npys NN_model_4000x4000
"""
import sys
import json
from pathlib import Path
import numpy as np
import torch

def load_one_input(npy_path: Path) -> torch.Tensor:
    arr = np.load(npy_path)
    if arr.ndim == 2:
        arr = arr[None, ...]  # (H,W) -> (1,H,W)
    elif arr.ndim != 3:
        raise ValueError(f"Expected (H,W) or (T,H,W); got {arr.shape}")
    x = torch.from_numpy(arr.astype(np.float32, copy=False))[None, ...]  # [1,C,H,W]
    return x

def maybe_list_inputs(source: Path):
    if source.is_dir():
        files = sorted(p for p in source.glob("*.npy") if p.is_file())
        if not files:
            raise FileNotFoundError(f"No .npy files in directory {source}")
        return files
    elif source.is_file():
        return [source]
    else:
        raise FileNotFoundError(source)

def load_predictor(artifact_dir: Path, in_channels: int):
    meta = json.loads((artifact_dir / "trained_model_meta.json").read_text())
    target_space = meta["target_space"]

    ts = artifact_dir / "predictor.ts"
    if ts.exists():
        print(f"Loading TorchScript: {ts}")
        model = torch.jit.load(str(ts), map_location="cpu")
        model.eval()
        return model, target_space, "ts"

    print("TorchScript not found; using state_dict + Python model.")
    from model import EnergyRegressionCNN
    weights = Path(meta.get("best_state_dict", artifact_dir / "best.pth"))
    model = EnergyRegressionCNN(in_channels=in_channels)
    sd = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, target_space, "pth"

def invert_target_space(y_hat_scalar: float, target_space: str) -> float:
    if target_space == "log":
        return float(np.exp(y_hat_scalar))
    return float(y_hat_scalar)

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    src = Path(sys.argv[1])
    outdir = Path(sys.argv[2])

    files = maybe_list_inputs(src)

    # Peek first file to get in_channels if we end up loading state_dict
    x0 = load_one_input(files[0])
    in_channels = x0.shape[1]

    model, target_space, kind = load_predictor(outdir, in_channels)
    print(f"Predictor kind: {kind}, target_space: {target_space}")

    preds = []
    with torch.no_grad():
        for f in files:
            x = load_one_input(f)
            y_hat = model(x).squeeze().item()
            pred_gev = invert_target_space(y_hat, target_space)
            preds.append((f.name, pred_gev))
            print(f"{f.name}: {pred_gev:.6f} GeV")

    # Optional: write a tiny CSV next to the outdir
    csv_path = outdir / "predictions.csv"
    try:
        with open(csv_path, "w") as fh:
            fh.write("file,pred_energy_GeV\n")
            for name, val in preds:
                fh.write(f"{name},{val}\n")
        print(f"📝 Wrote {csv_path}")
    except Exception:
        pass

if __name__ == "__main__":
    main()


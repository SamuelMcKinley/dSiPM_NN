#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from contextlib import nullcontext

from dataset import PhotonEnergyDataset
from model import EnergyRegressionCNN


def parse_args():
    p = argparse.ArgumentParser(description="Incremental training for EnergyRegressionCNN on photon tensors.")
    p.add_argument("tensor_path", help="Path to a single .npy OR a directory containing .npy files")
    p.add_argument("--energy", type=float,
                   help="True energy in GeV for these tensors (optional if filenames encode energies)")
    p.add_argument("--spad", required=True, help='SPAD size label (e.g., "50x50", "70x70", "4000x4000")')
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--target-space", choices=["linear", "log"], default="log",
                   help="Train to predict energy (linear) or log(energy) (log)")
    p.add_argument("--workers", type=int, default=8, help="Number of DataLoader workers")
    p.add_argument("--cpu-threads", type=int, default=0, help="torch.set_num_threads; 0 = leave default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="NN_model_{spad}",
                   help='Output directory (supports {spad} and {energy}). Default: "NN_model_{spad}"')
    p.add_argument("--val-split", type=float, default=0.2, help="Validation fraction (when >1 samples)")
    p.add_argument("--save-ts", action="store_true", help="Also save a TorchScript predictor")
    return p.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_outdir(args):
    outdir = Path(args.outdir.format(spad=args.spad, energy=args.energy)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def maybe_load_checkpoint(model, optimizer, ckpt_path: Path):
    """Load model/optimizer if checkpoint exists. Return (start_epoch, best_val)."""
    start_epoch, best_val = 0, float("inf")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"↩️  Resumed from {ckpt_path} (epoch={start_epoch}, best_val={best_val:.6f})")
    else:
        print("🆕 No checkpoint found — starting fresh.")
    return start_epoch, best_val


def save_checkpoint(ckpt_path: Path, model, optimizer, epoch, best_val, meta: dict):
    ckpt = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "meta": meta,
    }
    tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(ckpt_path)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    # ------- Dataset / Split -------
    dataset = PhotonEnergyDataset(
        tensor_path=args.tensor_path,
        energy=args.energy,  # may be None for mixed-energy folders
        target_space=args.target_space,
        mmap=True,
    )

    energies_in_data = dataset.get_all_energies()
    if len(set(energies_in_data)) > 1:
        print(f"🧠 Loaded mixed-energy dataset with {len(dataset)} samples:")
        print(f"   Energies found: {sorted(set(energies_in_data))}")
    else:
        print(f"🧠 Loaded dataset with {len(dataset)} samples at {energies_in_data[0]} GeV")

    # ------- Train/Val split -------
    if len(dataset) <= 1 or args.val_split <= 0:
        train_set = dataset
        val_set = dataset
    else:
        train_len = max(1, int(len(dataset) * (1.0 - args.val_split)))
        val_len = max(1, len(dataset) - train_len)
        generator = torch.Generator().manual_seed(args.seed)
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    # ------- Device / AMP -------
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    amp_ctx = torch.amp.autocast("cuda", enabled=use_cuda)
    scaler = torch.amp.GradScaler("cuda") if use_cuda else None

    # ------- DataLoaders -------
    pin = use_cuda
    workers = max(0, args.workers)
    train_loader = DataLoader(
        train_set, batch_size=args.bs, shuffle=True,
        num_workers=workers, pin_memory=pin, persistent_workers=(workers > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.bs, shuffle=False,
        num_workers=workers, pin_memory=pin, persistent_workers=(workers > 0),
        drop_last=False,
    )

    # ------- Model / Loss / Optim -------
    model = EnergyRegressionCNN(in_channels=dataset.channels).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ------- Outdir / Logging setup -------
    outdir = prepare_outdir(args)
    loss_log_path = (
        outdir / f"loss_history_{int(args.energy)}.txt"
        if args.energy is not None
        else outdir / "loss_history_mixed.txt"
    )
    last_ckpt = outdir / "last.ckpt"
    best_ckpt = outdir / "best.ckpt"
    best_state = outdir / "best.pth"
    meta_path = outdir / "trained_model_meta.json"

    start_epoch, best_val = maybe_load_checkpoint(model, optimizer, last_ckpt)

    # ------- Training loop -------
    total_epochs = args.epochs
    loss_log_file = open(loss_log_path, "a")

    for epoch in range(start_epoch + 1, start_epoch + total_epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                out = model(x)
                if out.ndim > 1 and out.size(-1) == 1:
                    out = out.squeeze(-1)
                loss = criterion(out, y)

            if use_cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss_sum += loss.item()

        avg_train = train_loss_sum / max(1, len(train_loader))

        # ------- Validation -------
        model.eval()
        val_loss_sum = 0.0
        preds_list, targets_list = [], []
        with torch.no_grad(), (amp_ctx if use_cuda else nullcontext()):
            for x, y in val_loader:
                x = x.to(device, non_blocking=use_cuda)
                y = y.to(device, non_blocking=use_cuda)
                out = model(x)
                if out.ndim > 1 and out.size(-1) == 1:
                    out = out.squeeze(-1)
                val_loss_sum += criterion(out, y).item()
                preds_list.append(out.detach().cpu())
                targets_list.append(y.detach().cpu())

        avg_val = val_loss_sum / max(1, len(val_loader))
        print(f"Epoch {epoch:02d}/{start_epoch + total_epochs} | Train {avg_train:.6f} | Val {avg_val:.6f}")

        # ------- Log metrics -------
        if args.target_space == "log":
            preds_linear = torch.exp(torch.cat(preds_list))
            targets_linear = torch.exp(torch.cat(targets_list))
        else:
            preds_linear = torch.cat(preds_list)
            targets_linear = torch.cat(targets_list)

        mean_pred = preds_linear.mean().item()
        mean_true = targets_linear.mean().item()

        loss_log_file.write(f"{epoch},{avg_train:.6f},{avg_val:.6f},{mean_pred:.6f},{mean_true:.6f}\n")
        loss_log_file.flush()

        # ------- Checkpointing -------
        meta = {
            "target_space": args.target_space,
            "energy_GeV": float(args.energy) if args.energy is not None else "mixed",
            "spad_size": args.spad,
            "best_val_loss": float(best_val),
            "batch_size": args.bs,
            "learning_rate": args.lr,
            "num_epochs_this_run": args.epochs,
            "workers": args.workers,
            "device": "cuda" if use_cuda else "cpu",
            "tensor_path": os.path.abspath(args.tensor_path),
            "seed": args.seed,
            "epoch": epoch,
        }
        save_checkpoint(last_ckpt, model, optimizer, epoch, best_val, meta)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_state)
            save_checkpoint(best_ckpt, model, optimizer, epoch, best_val, meta)
            print(f"💾 New best (val={best_val:.6f}) saved to {best_ckpt} and {best_state}")

    loss_log_file.close()

    # ------- Export TorchScript (optional) -------
    if args.save_ts:
        model.eval()
        C, H, W = dataset.channels, dataset.height, dataset.width
        example = torch.zeros(1, C, H, W, device=device, dtype=torch.float32)
        scripted = torch.jit.trace(model, example)
        ts_path = outdir / "predictor.ts"
        scripted.save(str(ts_path))
        print(f"🧠 TorchScript predictor saved to {ts_path}")

    # ------- Save metadata -------
    meta["best_val_loss"] = float(best_val)
    meta["best_ckpt"] = str(best_ckpt)
    meta["best_state_dict"] = str(best_state)
    meta["last_ckpt"] = str(last_ckpt)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Training finished. Best={best_val:.6f}. Artifacts in {outdir}")


if __name__ == "__main__":
    main()
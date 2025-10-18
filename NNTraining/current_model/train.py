#!/usr/bin/env python3
import os
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from contextlib import nullcontext

from dataset import PhotonEnergyDataset
from model import EnergyRegressionCNN


def parse_args():
    p = argparse.ArgumentParser(description="Train CNN to predict linear energy from photon tensors.")
    p.add_argument("tensor_path", help="Folder of .npy tensors (or a single .npy file)")
    p.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    p.add_argument("--bs", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--val-split", type=float, default=0.30, help="Validation fraction (default 0.30 = 30%)")
    p.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    p.add_argument("--cpu-threads", type=int, default=0, help="torch.set_num_threads; 0 = leave default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="NN_model", help="Output directory for runs")
    p.add_argument("--tag", default="", help="Optional tag to distinguish runs")
    p.add_argument("--early-stop", type=int, default=0, help="Early stop patience (epochs). 0=off")
    return p.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_outdir(base: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{base}"
    if tag:
        name += f"_{tag}"
    name += f"_{ts}"
    out = Path(name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: dict, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    # -------- Dataset --------
    dataset = PhotonEnergyDataset(args.tensor_path, mmap=True)
    uniq_e = sorted(set(dataset.get_all_energies()))
    print(f"Loaded {len(dataset)} samples. Energies present: {uniq_e}")

    # 70/30 split (as requested) using random_split with fixed seed
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * args.val_split)))
    n_train = max(1, n_total - n_val)

    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)

    # -------- Device / AMP --------
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

    # -------- Loaders --------
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

    # -------- Model / Loss / Optim --------
    in_channels = dataset.channels
    model = EnergyRegressionCNN(in_channels=in_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -------- Outputs --------
    outdir = make_outdir(args.outdir, args.tag)
    ckpt_last = outdir / "last.ckpt"
    ckpt_best = outdir / "best.ckpt"
    best_state = outdir / "best.pth"
    meta_path = outdir / "run_meta.json"
    loss_csv = outdir / "loss_history.csv"

    # Write loss CSV header
    with open(loss_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "train_loss", "val_loss",
            "val_mae", "val_rmse",
            "val_mean_true", "val_mean_pred",
            "num_train", "num_val",
        ])

    best_val = float("inf")
    no_improve = 0

    # -------- Training --------
    total_epochs = args.epochs
    for epoch in range(1, total_epochs + 1):
        # ----- Train -----
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            x, y, _names = batch  # names unused in training
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

        # ----- Validation -----
        model.eval()
        val_loss_sum = 0.0
        preds_all, trues_all, names_all = [], [], []

        with torch.no_grad(), (amp_ctx if use_cuda else nullcontext()):
            for x, y, names in val_loader:
                x = x.to(device, non_blocking=use_cuda)
                y = y.to(device, non_blocking=use_cuda)
                out = model(x)
                if out.ndim > 1 and out.size(-1) == 1:
                    out = out.squeeze(-1)

                batch_loss = criterion(out, y).item()
                val_loss_sum += batch_loss

                preds_all.append(out.detach().cpu())
                trues_all.append(y.detach().cpu())
                names_all.extend(list(names))

        import torch as _torch
        preds = _torch.cat(preds_all).float()
        trues = _torch.cat(trues_all).float()

        diffs = preds - trues
        val_mae = diffs.abs().mean().item()
        val_rmse = (diffs.pow(2).mean().sqrt().item())
        avg_val = val_loss_sum / max(1, len(val_loader))
        mean_pred = preds.mean().item()
        mean_true = trues.mean().item()

        print(
            f"Epoch {epoch:03d}/{total_epochs} | "
            f"Train {avg_train:.6f} | Val {avg_val:.6f} | "
            f"MAE {val_mae:.6f} | RMSE {val_rmse:.6f}"
        )

        # ----- Append epoch summary to CSV -----
        with open(loss_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, f"{avg_train:.6f}", f"{avg_val:.6f}",
                f"{val_mae:.6f}", f"{val_rmse:.6f}",
                f"{mean_true:.6f}", f"{mean_pred:.6f}",
                len(train_set), len(val_set),
            ])

        # ----- Write per-sample validation predictions for this epoch -----
        per_epoch_csv = outdir / f"val_predictions_epoch_{epoch:03d}.csv"
        with open(per_epoch_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "true_energy", "pred_energy", "deviation", "squared_error", "abs_error"])
            for name, t, p in zip(names_all, trues.tolist(), preds.tolist()):
                dev = p - t
                w.writerow([name, f"{t:.6f}", f"{p:.6f}", f"{dev:.6f}", f"{dev*dev:.6f}", f"{abs(dev):.6f}"])

        # ----- Checkpointing -----
        meta = {
            "device": "cuda" if use_cuda else "cpu",
            "epochs": args.epochs,
            "batch_size": args.bs,
            "learning_rate": args.lr,
            "val_split": args.val_split,
            "workers": args.workers,
            "seed": args.seed,
            "tensor_path": os.path.abspath(args.tensor_path),
            "in_channels": in_channels,
            "dataset_size": len(dataset),
            "train_size": len(train_set),
            "val_size": len(val_set),
            "energies_present": uniq_e,
            "epoch": epoch,
            "best_val": float(best_val),
        }

        # Save last
        torch.save(
            {"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch, "best_val": best_val, "meta": meta},
            ckpt_last,
        )

        # Save best
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_state)
            torch.save(
                {"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch, "best_val": best_val, "meta": meta},
                ckpt_best,
            )
            print(f"💾 New best (val={best_val:.6f}) saved to {ckpt_best} and {best_state}")
            no_improve = 0
        else:
            no_improve += 1

        # Early stopping
        if args.early_stop > 0 and no_improve >= args.early_stop:
            print(f"⏹️ Early stopping triggered (patience={args.early_stop}).")
            break

    # -------- Save final metadata --------
    final_meta = {
        "device": "cuda" if use_cuda else "cpu",
        "best_val": float(best_val),
        "loss_history_csv": str(loss_csv),
        "tensor_path": os.path.abspath(args.tensor_path),
        "outdir": str(outdir),
    }
    save_json(final_meta, meta_path)
    print(f"✅ Finished. Artifacts in: {outdir}")


if __name__ == "__main__":
    main()
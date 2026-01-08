#!/usr/bin/env python3
import os, csv, argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from contextlib import nullcontext

from dataset import PhotonCountEnergyDataset
from model import PhotonMLP

def parse_args():
    p = argparse.ArgumentParser(description="Train MLP: (nPhotons -> energy), resumable, balanced per energy.")
    p.add_argument("csv_path", help="Path to photon_energy_<spad>.csv")
    p.add_argument("--spad", required=True, help="SPAD label (e.g. 70x70)")
    p.add_argument("--group", default=None, help="Optional group label (e.g. G0)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-dir", default="NN_photons_model", help="Base output directory")
    p.add_argument("--early-stop", type=int, default=10)
    p.add_argument("--transform", default="log1p", choices=["log1p", "none"])
    p.add_argument("--hidden", default="32,32", help="Hidden sizes, comma-separated (e.g. 64,64,32)")
    p.add_argument("--dropout", type=float, default=0.0)
    return p.parse_args()

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_rows(loss_csv: Path):
    if not loss_csv.exists():
        return []
    with open(loss_csv, "r", newline="") as f:
        return list(csv.DictReader(f))

def determine_group(loss_csv: Path, requested: str | None) -> str:
    rows = read_rows(loss_csv)
    used = {r["group"] for r in rows} if rows else set()
    if requested:
        if requested not in used:
            return requested
        base = requested.rstrip("0123456789")
        i = 1
        while f"{base}{i}" in used:
            i += 1
        return f"{base}{i}"
    i = 0
    while f"G{i}" in used:
        i += 1
    return f"G{i}"

def next_cumulative_epoch(loss_csv: Path) -> int:
    rows = read_rows(loss_csv)
    if not rows:
        return 1
    mx = 0
    for r in rows:
        e = r.get("epoch", "")
        if str(e).isdigit():
            mx = max(mx, int(e))
    return mx + 1

def stratified_indices_by_energy(energies: List[float], val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    energies = np.asarray(energies)
    uniq = sorted(set(energies.tolist()))
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for E in uniq:
        idx = np.where(energies == E)[0].tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
        if n_val >= n and n > 1:
            n_val = n - 1
        va.extend(idx[:n_val])
        tr.extend(idx[n_val:])
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va

def compute_norm(x_train: np.ndarray):
    mu = float(np.mean(x_train))
    sigma = float(np.std(x_train))
    if sigma <= 0:
        sigma = 1.0
    return mu, sigma

def main():
    args = parse_args()
    set_seed(args.seed)

    outdir = Path(f"{args.base_dir}_{args.spad}").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    loss_csv = outdir / "loss_history.csv"
    val_csv  = outdir / "val_predictions_all_epochs.csv"
    if not loss_csv.exists():
        with open(loss_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "group","epoch","train_loss","val_loss","val_mae","val_rmse",
                "val_mean_true","val_mean_pred","num_train","num_val",
                "x_mu_train","x_sigma_train"
            ])
    if not val_csv.exists():
        with open(val_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "group","epoch","nPhotons_feat","true_energy","pred_energy",
                "deviation","squared_error","abs_error"
            ])

    group = determine_group(loss_csv, args.group)
    start_epoch = next_cumulative_epoch(loss_csv)
    print(f"Group {group} | Cumulative training will start at epoch {start_epoch}")

    ds = PhotonCountEnergyDataset(args.csv_path, transform=args.transform)
    all_E = ds.get_all_energies()
    print(f"[{group}] Loaded {len(ds)} rows | Energies present: {sorted(set(all_E))}")

    tr_idx, va_idx = stratified_indices_by_energy(all_E, args.val_split, args.seed)
    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise RuntimeError("Stratified split failed: need at least 1 sample in train and val.")

    train_set = Subset(ds, tr_idx)
    val_set   = Subset(ds, va_idx)

    use_cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    amp_ctx = torch.amp.autocast("cuda", enabled=use_cuda)
    scaler = torch.amp.GradScaler("cuda") if use_cuda else None

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=0, pin_memory=use_cuda)
    val_loader   = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=use_cuda)

    hidden = tuple(int(x.strip()) for x in args.hidden.split(",") if x.strip())
    model = PhotonMLP(hidden=hidden, dropout=args.dropout).to(dev)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5,
                                                     threshold=1e-4, min_lr=1e-6)

    best_latest = outdir / "best_latest.ckpt"
    if best_latest.exists():
        print(f"Resuming from {best_latest}")
        ckpt = torch.load(best_latest, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        try:
            optimizer.load_state_dict(ckpt["optim"])
        except Exception:
            pass
    else:
        print("No prior checkpoint — starting fresh weights.")

    # Feature normalization computed on TRAIN split only
    x_all = ds.get_all_features()
    x_train = x_all[np.array(tr_idx, dtype=int)]
    x_mu, x_sigma = compute_norm(x_train)
    print(f"[{group}] Feature norm (train only): mu={x_mu:.6f}, sigma={x_sigma:.6f}")

    best_val = float("inf")
    no_improve = 0

    cur_epoch = start_epoch
    for _ in range(args.epochs):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(dev, non_blocking=use_cuda)
            y = y.to(dev, non_blocking=use_cuda)

            # normalize feature
            x = (x - x_mu) / x_sigma

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                out = model(x).squeeze(-1)
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

        # ---- Validate ----
        model.eval()
        val_loss_sum = 0.0
        preds, trues, feats = [], [], []

        with torch.no_grad(), (amp_ctx if use_cuda else nullcontext()):
            for x, y in val_loader:
                x = x.to(dev, non_blocking=use_cuda)
                y = y.to(dev, non_blocking=use_cuda)

                x_norm = (x - x_mu) / x_sigma
                out = model(x_norm).squeeze(-1)

                val_loss_sum += criterion(out, y).item()

                preds.append(out.detach().cpu())
                trues.append(y.detach().cpu())
                feats.append(x.detach().cpu())  # stored as "feature" (log1p or raw)

        preds = torch.cat(preds).float()
        trues = torch.cat(trues).float()
        feats = torch.cat(feats).float().squeeze(-1)

        diffs = preds - trues
        val_mae = float(diffs.abs().mean().item())
        val_rmse = float(diffs.pow(2).mean().sqrt().item())
        avg_val = val_loss_sum / max(1, len(val_loader))
        mean_true = float(trues.mean().item())
        mean_pred = float(preds.mean().item())

        print(f"[{group}] Epoch {cur_epoch:03d} | Train {avg_train:.6f} | Val {avg_val:.6f} | "
              f"MAE {val_mae:.6f} | RMSE {val_rmse:.6f}")

        with open(loss_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                group, cur_epoch,
                f"{avg_train:.6f}", f"{avg_val:.6f}",
                f"{val_mae:.6f}", f"{val_rmse:.6f}",
                f"{mean_true:.6f}", f"{mean_pred:.6f}",
                len(train_set), len(val_set),
                f"{x_mu:.6f}", f"{x_sigma:.6f}",
            ])

        with open(val_csv, "a", newline="") as f:
            w = csv.writer(f)
            for feat, t, p in zip(feats.tolist(), trues.tolist(), preds.tolist()):
                d = p - t
                w.writerow([
                    group, cur_epoch,
                    f"{feat:.6f}", f"{t:.6f}", f"{p:.6f}",
                    f"{d:.6f}", f"{(d*d):.6f}", f"{abs(d):.6f}"
                ])

        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {"model": model.state_dict(), "optim": optimizer.state_dict(),
                 "best_val": best_val, "x_mu_train": x_mu, "x_sigma_train": x_sigma},
                best_latest
            )
            torch.save(model.state_dict(), outdir / "best_latest.pth")
            print(f"💾 New best (val={best_val:.6f}) saved to {best_latest}")
            no_improve = 0
        else:
            no_improve += 1

        if args.early_stop > 0 and no_improve >= args.early_stop:
            print(f"[{group}] Early stopping (patience={args.early_stop})")
            break

        cur_epoch += 1

    print(f"[{group}] Finished. Best val={best_val:.6f}. Logs in {outdir}")

if __name__ == "__main__":
    main()
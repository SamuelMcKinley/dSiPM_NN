#!/usr/bin/env python3
import os, csv, argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from contextlib import nullcontext

from dataset import PhotonEnergyDataset
from model import EnergyRegressionCNN


def parse_args():
    p = argparse.ArgumentParser(description="Train CNN to predict energy from hit-map tensors (resumable, balanced).")
    p.add_argument("tensor_path", help="Folder (or single .npy) of tensors")
    p.add_argument("--spad", required=True, help="SPAD size label (e.g. 4000x4000)")
    p.add_argument("--group", default=None, help="Optional group label (e.g. G7). If omitted, auto-assigns.")
    p.add_argument("--epochs", type=int, default=50, help="Epochs per run")
    p.add_argument("--bs", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    p.add_argument("--val-split", type=float, default=0.30, help="Validation fraction per-energy")
    p.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    p.add_argument("--cpu-threads", type=int, default=0, help="torch.set_num_threads; 0=leave default")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--base-dir", default="NN_model", help="Base output directory (shared across runs)")
    p.add_argument("--early-stop", type=int, default=0, help="Patience in epochs (0=off)")
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_loss_history(loss_csv: Path) -> List[Dict[str, str]]:
    if not loss_csv.exists():
        return []
    with open(loss_csv, "r", newline="") as f:
        return list(csv.DictReader(f))


def determine_group(loss_csv: Path, requested_group: str | None) -> str:
    rows = read_loss_history(loss_csv)
    used = {r["group"] for r in rows} if rows else set()

    if requested_group:
        if requested_group not in used:
            return requested_group
        base = requested_group.rstrip("0123456789")
        i = 1
        while f"{base}{i}" in used:
            i += 1
        return f"{base}{i}"

    i = 0
    while f"G{i}" in used:
        i += 1
    return f"G{i}"


def next_cumulative_epoch(loss_csv: Path) -> int:
    rows = read_loss_history(loss_csv)
    if not rows:
        return 1
    mx = 0
    for r in rows:
        e = r.get("epoch", "")
        if str(e).isdigit():
            mx = max(mx, int(e))
    return mx + 1


def stratified_indices_by_energy(energies: List[float], val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    energies = np.asarray(energies, dtype=np.float32)

    # Important: avoid float-equality surprises
    energies = np.round(energies, 6)

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

    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


def compute_target_norm(train_targets: torch.Tensor) -> Tuple[float, float]:
    mu = float(train_targets.mean().item())
    sigma = float(train_targets.std().item())
    if sigma <= 0:
        sigma = 1.0
    return mu, sigma


def normalize_targets(y: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    return (y - mu) / sigma


def denormalize_targets(y_norm: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    return y_norm * sigma + mu


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    outdir = Path(f"{args.base_dir}_{args.spad}").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    loss_csv = outdir / "loss_history.csv"
    val_csv  = outdir / "val_predictions_all_epochs.csv"

    if not loss_csv.exists():
        with open(loss_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "group","epoch","train_loss","val_loss","val_mae","val_rmse",
                "val_mean_true","val_mean_pred","num_train","num_val",
                "mu_train","sigma_train","best_val_at_save"
            ])

    if not val_csv.exists():
        with open(val_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "group","epoch","filename","true_energy","pred_energy",
                "deviation","squared_error","abs_error"
            ])

    group = determine_group(loss_csv, args.group)
    start_epoch = next_cumulative_epoch(loss_csv)
    print(f"Group {group} | Cumulative training will start at epoch {start_epoch}")

    ds = PhotonEnergyDataset(args.tensor_path)
    all_E = ds.get_all_energies()
    print(f"[{group}] Loaded {len(ds)} samples | Energies present: {sorted(set(all_E))}")

    tr_idx, va_idx = stratified_indices_by_energy(all_E, args.val_split, args.seed)
    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise RuntimeError("Stratified split failed: need at least 1 sample in train and val.")

    train_set = Subset(ds, tr_idx)
    val_set   = Subset(ds, va_idx)

    use_cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    amp_ctx = torch.amp.autocast("cuda", enabled=use_cuda)
    scaler = torch.amp.GradScaler("cuda") if use_cuda else None

    train_loader = DataLoader(
        train_set, batch_size=args.bs, shuffle=True,
        num_workers=max(0, args.workers), pin_memory=use_cuda, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=args.bs, shuffle=False,
        num_workers=max(0, args.workers), pin_memory=use_cuda, drop_last=False
    )

    model = EnergyRegressionCNN(in_channels=ds.channels).to(dev)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
        threshold=1e-4, cooldown=0, min_lr=1e-6
    )

    best_latest_ckpt = outdir / "best_latest.ckpt"
    best_latest_pth  = outdir / "best_latest.pth"

    # ---- Resume state (CRITICAL FIXES) ----
    best_val = float("inf")
    mu_train = None
    sigma_train = None

    if best_latest_ckpt.exists():
        print(f"Resuming from {best_latest_ckpt}")
        ckpt = torch.load(best_latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        try:
            optimizer.load_state_dict(ckpt["optim"])
        except Exception:
            pass

        best_val = float(ckpt.get("best_val", best_val))

        mu_train = ckpt.get("mu_train", None)
        sigma_train = ckpt.get("sigma_train", None)

        if mu_train is None or sigma_train is None:
            print("WARNING: checkpoint missing mu/sigma; will recompute once (not ideal).")

    else:
        print("No prior checkpoint — starting fresh weights.")

    # ---- Compute mu/sigma ONLY if we don't have them from ckpt ----
    if mu_train is None or sigma_train is None:
        train_targets_list = []
        with torch.no_grad():
            for _, y, _ in DataLoader(train_set, batch_size=512, shuffle=False, num_workers=0):
                train_targets_list.append(y)
        train_targets = torch.cat(train_targets_list).float()
        mu_train, sigma_train = compute_target_norm(train_targets)

    mu_train = float(mu_train)
    sigma_train = float(sigma_train)
    print(f"[{group}] Target normalization (fixed across resumes): mu={mu_train:.6f}, sigma={sigma_train:.6f}")
    print(f"[{group}] Loaded best_val from checkpoint: {best_val:.6f}" if best_latest_ckpt.exists() else "")

    no_improve = 0
    cur_epoch = start_epoch

    for _ in range(args.epochs):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0

        for x, y, _ in train_loader:
            x = x.to(dev, non_blocking=use_cuda)
            y = y.to(dev, non_blocking=use_cuda)

            y_norm = normalize_targets(y, mu_train, sigma_train)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                out_norm = model(x).squeeze(-1)
                loss = criterion(out_norm, y_norm)

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
        preds_denorm, trues_denorm, names_all = [], [], []

        with torch.no_grad(), (amp_ctx if use_cuda else nullcontext()):
            for x, y, names in val_loader:
                x = x.to(dev, non_blocking=use_cuda)
                y = y.to(dev, non_blocking=use_cuda)

                out_norm = model(x).squeeze(-1)
                y_norm = normalize_targets(y, mu_train, sigma_train)
                val_loss_sum += criterion(out_norm, y_norm).item()

                out_denorm = denormalize_targets(out_norm, mu_train, sigma_train)

                preds_denorm.append(out_denorm.detach().cpu())
                trues_denorm.append(y.detach().cpu())
                names_all.extend(list(names))

        preds = torch.cat(preds_denorm).float()
        trues = torch.cat(trues_denorm).float()

        diffs = preds - trues
        val_mae = float(diffs.abs().mean().item())                       # GeV
        val_rmse = float(diffs.pow(2).mean().sqrt().item())              # GeV
        avg_val = val_loss_sum / max(1, len(val_loader))                 # normalized MSE
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
                f"{mu_train:.6f}", f"{sigma_train:.6f}",
                f"{best_val:.6f}"
            ])

        with open(val_csv, "a", newline="") as f:
            w = csv.writer(f)
            for name, t, p in zip(names_all, trues.tolist(), preds.tolist()):
                d = p - t
                w.writerow([
                    group, cur_epoch, name,
                    f"{t:.6f}", f"{p:.6f}", f"{d:.6f}",
                    f"{(d*d):.6f}", f"{abs(d):.6f}"
                ])

        scheduler.step(avg_val)

        # ---- Save ONLY if truly improved (CRITICAL FIX) ----
        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "best_val": best_val,
                    "mu_train": mu_train,
                    "sigma_train": sigma_train,
                },
                best_latest_ckpt
            )
            torch.save(model.state_dict(), best_latest_pth)
            torch.save(model.state_dict(), outdir / f"best_{group}.pth")
            print(f"💾 New best (val={best_val:.6f}) saved to {best_latest_ckpt}")
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
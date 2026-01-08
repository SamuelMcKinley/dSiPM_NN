#!/usr/bin/env python3
import argparse, csv, os
import numpy as np
import matplotlib.pyplot as plt

def normal_pdf(x, mu, sigma):
    if sigma <= 0:
        return np.zeros_like(x)
    return (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("val_csv", help="Path to val_predictions_all_epochs.csv")
    ap.add_argument("--group", default=None, help="Optional group filter (e.g. G7)")
    ap.add_argument("--last", type=int, default=15, help="Use last N epochs (default 50)")
    ap.add_argument("--bins", type=int, default=40, help="Histogram bins")
    ap.add_argument("--min-count", type=int, default=1, help="Skip energies with too few points")
    ap.add_argument("--xlim", type=float, default=None, help="Symmetric x-limit for residuals")
    ap.add_argument("--outdir", default="residual_plots", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---------------- Read CSV ----------------
    rows = []
    with open(args.val_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if args.group is not None and str(r["group"]) != str(args.group):
                continue
            try:
                epoch = int(float(r["epoch"]))
                t = float(r["true_energy"])
                p = float(r["pred_energy"])
            except Exception:
                continue
            rows.append((epoch, t, p))

    if not rows:
        raise RuntimeError("No valid rows found.")

    # ---------------- Select last epochs ----------------
    epochs = sorted({e for (e, _, _) in rows})
    last_epochs = epochs[-args.last:] if len(epochs) >= args.last else epochs
    last_set = set(last_epochs)

    rows = [(e, t, p) for (e, t, p) in rows if e in last_set]
    print(f"Using epochs: {last_epochs}")

    # ---------------- Group by energy ----------------
    by_energy = {}
    for _, t, p in rows:
        by_energy.setdefault(t, []).append(p - t)

    # ---------------- Plot per energy ----------------
    for E in sorted(by_energy.keys()):
        residuals = np.asarray(by_energy[E], dtype=float)

        if residuals.size < args.min_count:
            print(f"Skipping E={E}: only {residuals.size} points")
            continue

        mu = float(np.mean(residuals))
        sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0

        plt.figure(figsize=(6, 4))
        counts, edges, _ = plt.hist(
            residuals,
            bins=args.bins,
            density=True,
            alpha=0.7
        )

        x = np.linspace(edges[0], edges[-1], 400)
        plt.plot(x, normal_pdf(x, mu, sigma), linewidth=2)

        plt.axvline(0.0, linestyle="--", linewidth=1)
        plt.xlabel(r"$E_{\mathrm{pred}} - E_{\mathrm{true}}$")
        plt.ylabel("Density")
        plt.title(
            f"E = {E}  |  last {len(last_epochs)} epochs\n"
            f"$\\mu$ = {mu:.3f},  $\\sigma$ = {sigma:.3f}"
        )

        if args.xlim is not None:
            plt.xlim(-args.xlim, args.xlim)

        plt.tight_layout()

        fname = os.path.join(
            args.outdir,
            f"residual_E{E}_last{len(last_epochs)}epochs.png"
        )
        plt.savefig(fname, dpi=150)
        plt.close()

        print(f"Saved {fname}")

if __name__ == "__main__":
    main()
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
    ap.add_argument("--group", default=None)
    ap.add_argument("--last", type=int, default=5)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--xlim", type=float, default=None)
    args = ap.parse_args()

    rows = []
    with open(args.val_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if args.group is not None and str(row.get("group","")) != str(args.group):
                continue
            try:
                epoch = int(float(row["epoch"]))
                t = float(row["true_energy"])
                p = float(row["pred_energy"])
            except Exception:
                continue
            rows.append((epoch, t, p))

    if not rows:
        raise SystemExit("No rows found.")

    epochs = sorted({e for (e, _, _) in rows})
    last_epochs = epochs[-args.last:] if len(epochs) >= args.last else epochs
    last_set = set(last_epochs)
    rows = [(e, t, p) for (e, t, p) in rows if e in last_set]
    print(f"Using epochs: {last_epochs}")

    # Group residuals by energy
    by_E = {}
    pred_true = []
    for _, t, p in rows:
        by_E.setdefault(t, []).append(p - t)
        pred_true.append((t, p))

    os.makedirs("plots", exist_ok=True)
    os.makedirs("residual_plots", exist_ok=True)

    # pred vs true
    t_arr = np.array([x[0] for x in pred_true], dtype=float)
    p_arr = np.array([x[1] for x in pred_true], dtype=float)

    plt.figure()
    plt.scatter(t_arr, p_arr, s=12)
    mn, mx = float(np.min(t_arr)), float(np.max(t_arr))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True Energy")
    plt.ylabel("Pred Energy")
    plt.title(f"Pred vs True (last {len(last_epochs)} epochs)")
    plt.tight_layout()
    plt.savefig("plots/pred_vs_true.png", dpi=150)
    plt.close()

    # residuals vs true
    res = p_arr - t_arr
    plt.figure()
    plt.scatter(t_arr, res, s=12)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("True Energy")
    plt.ylabel("Residual (Pred - True)")
    plt.title(f"Residuals vs True (last {len(last_epochs)} epochs)")
    plt.tight_layout()
    plt.savefig("plots/residuals_vs_true.png", dpi=150)
    plt.close()

    # per-energy residual histograms
    for E in sorted(by_E.keys()):
        rvals = np.asarray(by_E[E], dtype=float)
        if rvals.size < args.min_count:
            print(f"Skipping E={E}: only {rvals.size} points")
            continue

        mu = float(np.mean(rvals))
        sigma = float(np.std(rvals, ddof=1)) if rvals.size > 1 else 0.0

        plt.figure(figsize=(6, 4))
        counts, edges, _ = plt.hist(rvals, bins=args.bins, density=True, alpha=0.7)
        x = np.linspace(edges[0], edges[-1], 400)
        plt.plot(x, normal_pdf(x, mu, sigma), linewidth=2)
        plt.axvline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Pred - True")
        plt.ylabel("Density")
        plt.title(f"E={E}  (last {len(last_epochs)} epochs)  mu={mu:.3f}, sigma={sigma:.3f}")
        if args.xlim is not None:
            plt.xlim(-args.xlim, args.xlim)
        plt.tight_layout()
        fname = os.path.join("residual_plots", f"residual_E{E}_last{len(last_epochs)}epochs.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import csv
import math
import numpy as np
import matplotlib.pyplot as plt


def mad_core_mask(x: np.ndarray, nsigma: float = 3.0) -> np.ndarray:
    if x.size == 0:
        return np.zeros(0, dtype=bool)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0.0:
        return np.ones_like(x, dtype=bool)
    sigma_robust = 1.4826 * mad
    return np.abs(x - med) <= nsigma * sigma_robust


def parse_args():
    ap = argparse.ArgumentParser(
        description="Energy resolution plot from val_predictions_all_epochs.csv using multiple steps (no pandas)"
    )
    ap.add_argument("--csv", default="val_predictions_all_epochs.csv", help="Input CSV file")
    ap.add_argument("--out", default="energy_resolution.png", help="Output PNG file")

    ap.add_argument("--emin", type=float, default=10.0, help="Min energy (GeV)")
    ap.add_argument("--emax", type=float, default=100.0, help="Max energy (GeV)")
    ap.add_argument("--estep", type=float, default=10.0, help="Energy step (GeV)")

    ap.add_argument("--core_sigma", type=float, default=3.0, help="MAD-core cut in robust-sigma units")
    ap.add_argument("--n_per_energy", type=int, default=200, help="Target number of events per energy")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed for reproducibility")

    ap.add_argument("--dedup_by_file", action="store_true",
                    help="If set, count each tensor filename at most once per energy (recommended).")

    ap.add_argument("--max_steps_back", type=int, default=None,
                    help="Optional limit on how many steps back from latest to scan (default: no limit).")

    ap.add_argument("--save_summary", default="energy_resolution_summary.csv",
                    help="Write summary CSV. Set to '' to disable.")

    return ap.parse_args()


def is_complete(counts, energies, target):
    for E in energies:
        if counts.get(E, 0) < target:
            return False
    return True


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    energies = np.arange(args.emin, args.emax + 0.5 * args.estep, args.estep)
    energies = [float(E) for E in energies]

    # --- Pass 1: determine latest step ---
    max_step = None
    with open(args.csv, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            try:
                step = int(row[1])
            except Exception:
                continue
            if (max_step is None) or (step > max_step):
                max_step = step

    if max_step is None:
        raise RuntimeError(f"Could not find any valid rows in {args.csv}")

    # We'll scan steps from max_step downward
    min_step_allowed = 0
    if args.max_steps_back is not None:
        min_step_allowed = max_step - int(args.max_steps_back)

    # --- Pass 2: collect events across steps (latest -> older) ---
    # We'll store candidates per energy as list of frac residuals.
    # If dedup_by_file: also track seen filenames per energy.
    cand = {E: [] for E in energies}
    counts = {E: 0 for E in energies}
    seen_files = {E: set() for E in energies} if args.dedup_by_file else None

    steps_used = set()
    total_rows_used = 0

    # To do "latest -> older" in one file pass, we can:
    # 1) load rows grouped by step (lightweight: store only needed columns), OR
    # 2) read entire file and bucket by step (safe for typical sizes).
    #
    # Your val file is at most: (#steps) * (val_events_per_step).
    # Even if 1000 steps * 200 events = 200k rows, that's fine.
    #
    # We'll bucket by step to allow deterministic newest->oldest scan.
    by_step = {}  # step -> list of (true_E, pred_E, filename)
    with open(args.csv, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            try:
                step = int(row[1])
                fname = row[2]
                true_E = float(row[4])
                pred_E = float(row[5])
            except Exception:
                continue

            # only keep energies we care about
            # float-safe match
            matched_E = None
            for E in energies:
                if math.isclose(true_E, E, rel_tol=0.0, abs_tol=1e-6):
                    matched_E = E
                    break
            if matched_E is None:
                continue

            if step < min_step_allowed:
                continue

            by_step.setdefault(step, []).append((matched_E, pred_E, fname))

    # Scan from newest to oldest
    for step in range(max_step, min_step_allowed - 1, -1):
        if is_complete(counts, energies, args.n_per_energy):
            break
        if step not in by_step:
            continue

        # Shuffle within the step so you don't always take the same ordering
        rows = by_step[step]
        rng.shuffle(rows)

        took_any = False
        for (E, pred_E, fname) in rows:
            if counts[E] >= args.n_per_energy:
                continue

            if args.dedup_by_file:
                if fname in seen_files[E]:
                    continue
                seen_files[E].add(fname)

            frac = (pred_E - E) / E
            cand[E].append(frac)
            counts[E] += 1
            total_rows_used += 1
            took_any = True

            if is_complete(counts, energies, args.n_per_energy):
                break

        if took_any:
            steps_used.add(step)

    # --- Compute resolution per energy ---
    summary = []  # (E, N_used_after_core, bias, sigma_core, rms_core, N_collected, steps_span)

    # steps span info
    if steps_used:
        newest = max(steps_used)
        oldest = min(steps_used)
        steps_span = f"{oldest}..{newest} ({len(steps_used)} steps)"
    else:
        steps_span = "none"

    for E in energies:
        frac = np.array(cand[E], dtype=float)
        n_collected = int(frac.size)

        if n_collected == 0:
            summary.append((E, 0, float("nan"), float("nan"), float("nan"), 0))
            continue

        mask = mad_core_mask(frac, nsigma=args.core_sigma)
        frac_core = frac[mask]
        if frac_core.size == 0:
            frac_core = frac

        bias = float(np.mean(frac_core))
        sigma = float(np.std(frac_core, ddof=1)) if frac_core.size > 1 else 0.0
        rms = float(np.sqrt(np.mean(frac_core ** 2)))

        summary.append((E, int(frac_core.size), bias, sigma, rms, n_collected))

    # # --- Save summary CSV ---
    # if args.save_summary:
    #     with open(args.save_summary, "w", newline="") as f:
    #         w = csv.writer(f)
    #         w.writerow([
    #             "E_true_GeV",
    #             "N_used_after_core",
    #             "bias_mean_frac",
    #             "sigma_frac_core",
    #             "rms_frac_core",
    #             "N_collected_before_core",
    #             "steps_used_span"
    #         ])
    #         for (E, Nused, bias, sigma, rms, ncol) in summary:
    #             w.writerow([
    #                 f"{E:.6f}",
    #                 Nused,
    #                 f"{bias:.12f}" if np.isfinite(bias) else "",
    #                 f"{sigma:.12f}" if np.isfinite(sigma) else "",
    #                 f"{rms:.12f}" if np.isfinite(rms) else "",
    #                 ncol,
    #                 steps_span
    #             ])

    # --- Plot ---
    x = np.array([s[0] for s in summary], dtype=float)
    y = np.array([s[3] for s in summary], dtype=float)  # sigma_frac_core
    n_used = np.array([s[1] for s in summary], dtype=int)
    n_col = np.array([s[5] for s in summary], dtype=int)

    plt.figure()
    plt.plot(x, y, marker="o", linestyle="-")
    plt.xlabel("True Energy E (GeV)")
    plt.ylabel("Core resolution σ[(Epred−Etrue)/Etrue]")
    title = f"Energy Resolution (MAD core {args.core_sigma}σ), target n={args.n_per_energy}"
    if args.dedup_by_file:
        title += ", dedup_by_file"
    plt.title(title)
    plt.grid(True)

    # annotate with N_used_after_core / N_collected
    # for xi, yi, nu, nc in zip(x, y, n_used, n_col):
    #     if np.isfinite(yi) and nc > 0:
    #         plt.annotate(f"{nu}/{nc}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    # --- Print summary ---
    print(f"[OK] Latest step in file: {max_step}")
    if args.max_steps_back is not None:
        print(f"[OK] Scanned steps: {min_step_allowed}..{max_step} (max_steps_back={args.max_steps_back})")
    else:
        print(f"[OK] Scanned steps: 0..{max_step} (no limit)")

    print(f"[OK] Target per energy: n_per_energy={args.n_per_energy}")
    print(f"[OK] Steps used: {steps_span}")
    print(f"[OK] Total rows used: {total_rows_used}")
    print(f"[OK] Wrote plot: {args.out}")
    if args.save_summary:
        print(f"[OK] Wrote summary: {args.save_summary}")
    print()
    print(" E(GeV)  N_collected  N_used   bias(mean frac)   sigma_frac(core)   rms_frac(core)")
    for (E, Nused, bias, sigma, rms, ncol) in summary:
        if ncol == 0:
            print(f"{E:6.0f} {ncol:11d} {Nused:7d} {'':>16} {'':>16} {'':>16}")
        else:
            print(f"{E:6.0f} {ncol:11d} {Nused:7d} {bias:16.6e} {sigma:16.6e} {rms:16.6e}")


if __name__ == "__main__":
    main()
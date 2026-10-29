"""
Monte Carlo analysis of LMS vs RLS (Numba + multiprocessing).

Run from project root:   python -m src.test_mc_njit

Start with a small N_RUNS (e.g. 20) to check timing, then raise it for the final run.
Results saved to data/raw/mc_raw_results.csv and results/tables/mc_summary.csv
"""
import os
import time
import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Pool, cpu_count

from .adaptive_filters_njit import lms_filter_njit, rls_filter_njit

# ================= Settings (edit these) =================
N_RUNS = 20                       # runs per SNR level  -> raise to 200+ for final run
N_SAMPLES = 20000                 # samples per run
M = 15                            # filter length
MU = 1e-3                         # LMS step size
LAM = 0.999                       # RLS forgetting factor (keep >= 0.995 on stationary data)
DELTA = 1.0                       # RLS init
SNR_DB_LIST = [0, 5, 10, 15, 20, 25, 30]
N_PROCS = max(1, cpu_count() - 1) # leave one thread free for the OS
TRUE_TAPS = np.array([1.0, 0.6, -0.4, 0.25, -0.1] + [0.0] * (M - 5))
OUT_RAW = "data/raw/mc_raw_results.csv"
OUT_SUMMARY = "results/tables/mc_summary.csv"
# =========================================================


def warm_up():
    """Force JIT compilation once, so timing reflects real per-trial cost."""
    z = np.zeros(50)
    lms_filter_njit(z, z, 4, 0.01)
    rls_filter_njit(z, z, 4, 0.999, 1.0)


def run_trial(args):
    snr_idx, snr_db, run = args
    rng = np.random.default_rng(snr_idx * 1_000_000 + run)   # deterministic, unique per trial

    x = rng.standard_normal(N_SAMPLES)
    d_clean = np.convolve(x, TRUE_TAPS, mode='full')[:N_SAMPLES]
    noise_power = np.mean(d_clean ** 2) / (10 ** (snr_db / 10))
    d = d_clean + np.sqrt(noise_power) * rng.standard_normal(N_SAMPLES)

    _, _, w_lms = lms_filter_njit(x, d, M, MU)
    _, _, w_rls = rls_filter_njit(x, d, M, LAM, DELTA)

    lms_err = np.linalg.norm(w_lms - TRUE_TAPS)
    rls_err = np.linalg.norm(w_rls - TRUE_TAPS)
    return {'snr_db': snr_db, 'run': run,
            'lms_err': lms_err, 'rls_err': rls_err,
            'diff': lms_err - rls_err}      # positive => RLS better


def main():
    os.makedirs(os.path.dirname(OUT_RAW), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)

    warm_up()
    jobs = [(i, snr, r) for i, snr in enumerate(SNR_DB_LIST) for r in range(N_RUNS)]
    print(f"{len(jobs)} trials on {N_PROCS} processes ...")

    t0 = time.time()
    with Pool(N_PROCS, initializer=warm_up) as pool:
        rows = pool.map(run_trial, jobs, chunksize=max(1, len(jobs) // (N_PROCS * 8)))
    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    print(f"Done in {elapsed:.2f} s  ({elapsed / len(df) * 1000:.2f} ms/trial wall-clock)\n")

    summary = []
    for snr_db, g in df.groupby('snr_db'):
        n = len(g)
        diff = g['diff'].values
        _, pval = stats.ttest_rel(g['lms_err'].values, g['rls_err'].values)
        summary.append({
            'snr_db': snr_db,
            'lms_mean': g['lms_err'].mean(), 'lms_std': g['lms_err'].std(ddof=1),
            'rls_mean': g['rls_err'].mean(), 'rls_std': g['rls_err'].std(ddof=1),
            'diff_mean': diff.mean(),
            'diff_ci95': 1.96 * diff.std(ddof=1) / np.sqrt(n),
            'p_value': pval, 'significant': pval < 0.05,
        })
    sdf = pd.DataFrame(summary)
    pd.set_option('display.float_format', lambda v: f'{v:.5f}')
    print(sdf.to_string(index=False))

    df.to_csv(OUT_RAW, index=False)
    sdf.to_csv(OUT_SUMMARY, index=False)
    print(f"\nSaved {OUT_RAW} and {OUT_SUMMARY}")


if __name__ == "__main__":
    main()

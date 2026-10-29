"""
test_mc.py — proper Monte Carlo analysis of LMS vs RLS filters.

Sweeps across SNR levels, runs N_RUNS trials per level, and computes
paired statistics (mean, std, 95% CI, paired t-test) comparing LMS and
RLS weight-identification error on a known FIR channel.

Adjust N_RUNS at the top: start small (e.g. 20) to sanity-check timing,
then bump up for the final run (e.g. 200+).
"""

import time
import numpy as np
import pandas as pd
from scipy import stats

try:
    from .adaptive_filters import lms_filter, rls_filter
except ImportError:
    from adaptive_filters import lms_filter, rls_filter

# ================= Settings (edit these) =================
N_RUNS = 20                 # runs per SNR level — bump up for final run (e.g. 200)
N_SAMPLES = 20000            # samples per run (set closer to real pipeline scale)
M = 15                        # filter length
MU = 1e-3                     # LMS step size
LAM = 0.999                   # RLS forgetting factor
SNR_DB_LIST = [0, 5, 10, 15, 20, 25, 30]   # sweep grid
TRUE_TAPS = np.array([1.0, 0.6, -0.4, 0.25, -0.1] + [0.0]*(M-5))  # known test channel, padded to M taps

# ================= Run sweep =================
rows = []
t_start = time.time()

for snr_db in SNR_DB_LIST:
    for run in range(N_RUNS):
        seed = hash((snr_db, run)) % (2**32)
        rng = np.random.default_rng(seed)

        x = rng.standard_normal(N_SAMPLES)
        d_clean = np.convolve(x, TRUE_TAPS, mode='full')[:N_SAMPLES]

        sig_power = np.mean(d_clean**2)
        noise_power = sig_power / (10**(snr_db/10))
        d = d_clean + np.sqrt(noise_power) * rng.standard_normal(N_SAMPLES)

        _, _, w_lms = lms_filter(x, d, M, MU)
        _, _, w_rls = rls_filter(x, d, M, LAM)

        lms_err = np.linalg.norm(w_lms - TRUE_TAPS)
        rls_err = np.linalg.norm(w_rls - TRUE_TAPS)

        rows.append({
            'snr_db': snr_db, 'run': run,
            'lms_err': lms_err, 'rls_err': rls_err,
            'diff': lms_err - rls_err   # paired difference: positive means RLS better
        })

elapsed = time.time() - t_start
df = pd.DataFrame(rows)
print(f"Completed {len(df)} trials ({len(SNR_DB_LIST)} SNR levels x {N_RUNS} runs) in {elapsed:.2f} sec")
print(f"Avg time per trial: {elapsed/len(df)*1000:.2f} ms\n")

# ================= Aggregate stats per SNR level =================
summary = []
for snr_db, group in df.groupby('snr_db'):
    lms = group['lms_err'].values
    rls = group['rls_err'].values
    diff = group['diff'].values

    lms_mean, lms_std = lms.mean(), lms.std(ddof=1)
    rls_mean, rls_std = rls.mean(), rls.std(ddof=1)

    n = len(diff)
    diff_mean = diff.mean()
    diff_sem = diff.std(ddof=1) / np.sqrt(n)
    ci95 = 1.96 * diff_sem

    # paired t-test: is RLS significantly better than LMS at this SNR?
    tstat, pval = stats.ttest_rel(lms, rls)

    summary.append({
        'snr_db': snr_db,
        'lms_mean': lms_mean, 'lms_std': lms_std,
        'rls_mean': rls_mean, 'rls_std': rls_std,
        'diff_mean': diff_mean, 'diff_ci95': ci95,
        'p_value': pval,
        'significant': pval < 0.05
    })

summary_df = pd.DataFrame(summary)
pd.set_option('display.float_format', lambda x: f'{x:.5f}')
print(summary_df.to_string(index=False))

# ================= Save results =================
df.to_csv('mc_raw_results.csv', index=False)
summary_df.to_csv('mc_summary.csv', index=False)
print("\nSaved raw results to mc_raw_results.csv")
print("Saved summary stats to mc_summary.csv")

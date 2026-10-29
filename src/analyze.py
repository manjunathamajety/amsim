"""
Analysis of the Monte Carlo sweep.  Run from the project root:

    python -m src.analyze                       # reads data/raw/full/*.csv
    python -m src.analyze data/raw/pilot.csv    # or any csv / folder of csvs

Outputs:  results/tables/*.csv   and   results/figures/*.png
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

FC, FS = 20_000, 400_000          # must match trial.py (kept here so analysis needs no numba)
METHODS = ['env', 'sync', 'lms', 'rls']
LABEL = {'env': 'Envelope', 'sync': 'Sync (phase-aligned)', 'lms': 'LMS carrier', 'rls': 'RLS carrier'}
COLOR = {'env': 'tab:red', 'sync': 'tab:blue', 'lms': 'tab:green', 'rls': 'tab:orange'}
OUT_FIG, OUT_TAB = 'results/figures', 'results/tables'


# ---------------------------------------------------------------- helpers
def nearest(values, v):
    values = np.asarray(sorted(set(values)))
    return values[np.argmin(np.abs(values - v))]


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (controls false discovery rate over the whole grid)."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def carrier_gain(delay, alpha):
    """|H(fc)| of the 2-path channel 1 + alpha*exp(-j 2 pi fc d / fs)."""
    return np.abs(1 + alpha * np.exp(-2j * np.pi * FC * np.asarray(delay) / FS))


# ---------------------------------------------------------------- load + integrity
def load(path):
    files = [path] if path.endswith('.csv') else sorted(glob.glob(os.path.join(path, '*.csv')))
    if not files:
        sys.exit(f"No csv files found in {path}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True), files


def integrity(df, files):
    print("=" * 64)
    print("1. DATA INTEGRITY")
    print("=" * 64)
    d, a, s = df.delay.nunique(), df.alpha.nunique(), df.snr_db.nunique()
    per = df.groupby(['delay', 'alpha', 'snr_db']).size()
    print(f"files: {len(files)}   rows: {len(df)}")
    print(f"grid: {d} delays x {a} alphas x {s} SNRs = {d*a*s} points; runs per point: min {per.min()}, max {per.max()}")
    ok = True
    if per.min() != per.max() or len(per) != d * a * s:
        print("  WARNING: grid is incomplete or uneven (missing or partial blocks?)"); ok = False
    nan = df[[f'nmse_{m}' for m in METHODS]].isna().sum().sum()
    print(f"NaN values in nmse columns: {nan}")
    if nan: ok = False
    if 'seed' in df:
        dup = df.seed.duplicated().sum()
        print(f"duplicate seeds: {dup}")
        if dup: ok = False
    print("integrity:", "OK" if ok else "CHECK THE WARNINGS ABOVE")
    print()


# ---------------------------------------------------------------- per-point statistics
def point_table(df):
    rows = []
    for (d, a, s), g in df.groupby(['delay', 'alpha', 'snr_db']):
        n = len(g)
        rec = {'delay': d, 'alpha': a, 'snr_db': s, 'n': n, 'carrier_gain': float(carrier_gain(d, a))}
        for m in METHODS:
            x = g[f'nmse_{m}'].values
            rec[f'mean_{m}'] = x.mean()
            rec[f'ci_{m}'] = 1.96 * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        ref = np.log10(np.clip(g.nmse_sync.values, 1e-12, None))
        for m in ['lms', 'rls']:
            dd = np.log10(np.clip(g[f'nmse_{m}'].values, 1e-12, None)) - ref      # paired, per trial
            rec[f'logratio_{m}'] = dd.mean()                                      # >0: adaptive worse
            rec[f'p_{m}'] = stats.ttest_1samp(dd, 0.0).pvalue if (n > 1 and dd.std(ddof=1) > 0) else 1.0
        rows.append(rec)
    t = pd.DataFrame(rows)
    for m in ['lms', 'rls']:
        t[f'sig_{m}'] = bh_adjust(t[f'p_{m}'].values) < 0.05
    return t


# ---------------------------------------------------------------- headline numbers
def headline(df, t):
    print("=" * 64)
    print("2. HEADLINE RESULTS")
    print("=" * 64)
    cols = [f'nmse_{m}' for m in METHODS]
    print("Mean NMSE over the whole grid:")
    print(df[cols].mean().rename(lambda c: LABEL[c[5:]]).round(4).to_string(), "\n")

    n_pts = len(t)
    for m in ['lms']:
        better = ((t[f'logratio_{m}'] < 0) & t[f'sig_{m}']).sum()
        worse = ((t[f'logratio_{m}'] > 0) & t[f'sig_{m}']).sum()
        print(f"{LABEL[m]} vs sync, paired test on log-NMSE ratio at each of {n_pts} grid points (BH-FDR 5%):")
        print(f"   adaptive significantly BETTER: {better} ({better/n_pts:.1%})")
        print(f"   adaptive significantly WORSE:  {worse} ({worse/n_pts:.1%})")
        print(f"   no significant difference:     {n_pts-better-worse} ({(n_pts-better-worse)/n_pts:.1%})")
        print(f"   typical ratio (geometric mean of NMSE_lms / NMSE_sync): {10**t[f'logratio_{m}'].mean():.2f}x\n")

    # boundary table: performance vs carrier gain |H(fc)|  (the research question)
    df = df.copy()
    df['gain'] = carrier_gain(df.delay.values, df.alpha.values)
    edges = [0, 0.1, 0.3, 0.6, 1.0, 2.01]
    labels = ['<0.1 (carrier ~nulled)', '0.1-0.3', '0.3-0.6', '0.6-1.0', '>1.0 (carrier boosted)']
    df['gain_bin'] = pd.cut(df.gain, edges, labels=labels, include_lowest=True)
    df['lr_lms'] = np.log10(np.clip(df.nmse_lms, 1e-12, None)) - np.log10(np.clip(df.nmse_sync, 1e-12, None))
    tab = df.groupby('gain_bin', observed=True).agg(
        trials=('nmse_sync', 'size'),
        env=('nmse_env', 'mean'), sync=('nmse_sync', 'mean'),
        lms=('nmse_lms', 'mean'), rls=('nmse_rls', 'mean'), lr=('lr_lms', 'mean'))
    tab['lms_over_sync'] = 10 ** tab.pop('lr')
    print("Mean NMSE by carrier gain |H(fc)| (all SNRs):")
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print(tab.to_string(), "\n")
    tab.to_csv(os.path.join(OUT_TAB, 'headline_by_carrier_gain.csv'))


# ---------------------------------------------------------------- figures
def fig_nmse_vs_snr(t):
    channels = [("mild fade", 3, 0.5), ("sideband null", 9, 1.0),
                ("deep carrier fade", 10, 0.9), ("carrier null", 10, 1.0)]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (name, d, a) in zip(axs.ravel(), channels):
        d, a = nearest(t.delay, d), nearest(t.alpha, a)
        g = t[(t.delay == d) & (t.alpha == a)].sort_values('snr_db')
        for m in METHODS:
            ax.errorbar(g.snr_db, g[f'mean_{m}'], yerr=g[f'ci_{m}'], marker='o', ms=3,
                        capsize=2, label=LABEL[m], color=COLOR[m])
        ax.set_yscale('log'); ax.grid(alpha=0.3)
        ax.set_title(f"{name}  (delay {d}, alpha {a}, |H(fc)|={carrier_gain(d, a):.2f})")
    for ax in axs[1]: ax.set_xlabel('SNR (dB)')
    for ax in axs[:, 0]: ax.set_ylabel('NMSE (mean, 95% CI)')
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_FIG, 'fig1_nmse_vs_snr.png'), dpi=140); plt.close(fig)


def _heat(ax, M, rows, cols, **kw):
    im = ax.imshow(M, aspect='auto', origin='lower', **kw)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f'{c:g}' for c in cols], fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{r:g}' for r in rows], fontsize=7)
    ax.set_xlabel('path-2 gain alpha'); return im


def fig_heatmaps(t, snr):
    g = t[t.snr_db == snr]
    rows, cols = sorted(g.delay.unique()), sorted(g.alpha.unique())
    mats = {m: g.pivot(index='delay', columns='alpha', values=f'mean_{m}').loc[rows, cols].values for m in METHODS}
    vmin = max(1e-4, min(np.nanmin(M) for M in mats.values()))
    vmax = max(np.nanmax(M) for M in mats.values())
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, m in zip(axs, METHODS):
        im = _heat(ax, mats[m], rows, cols, cmap='viridis_r', norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(LABEL[m])
    axs[0].set_ylabel('path-2 delay (samples)')
    fig.colorbar(im, ax=axs, label='mean NMSE (log scale)', shrink=0.9)
    fig.suptitle(f'NMSE over the channel grid at SNR = {snr:g} dB')
    fig.savefig(os.path.join(OUT_FIG, 'fig2_heatmaps.png'), dpi=140, bbox_inches='tight'); plt.close(fig)


def fig_ratio(t, snr):
    g = t[t.snr_db == snr]
    rows, cols = sorted(g.delay.unique()), sorted(g.alpha.unique())
    L = g.pivot(index='delay', columns='alpha', values='logratio_lms').loc[rows, cols].values
    S = g.pivot(index='delay', columns='alpha', values='sig_lms').loc[rows, cols].values.astype(bool)
    lim = max(0.05, np.nanmax(np.abs(L)))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = _heat(ax, L, rows, cols, cmap='RdBu_r', vmin=-lim, vmax=lim)
    yy, xx = np.where(S); ax.plot(xx, yy, 'k.', ms=4)
    ax.set_ylabel('path-2 delay (samples)')
    ax.set_title(f'log10(NMSE_LMS / NMSE_sync), SNR = {snr:g} dB\nred: adaptive worse, blue: adaptive better, dot: significant')
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, 'fig3_ratio_heatmap.png'), dpi=140); plt.close(fig)


def fig_null_sweep(t, snr):
    a = nearest(t.alpha, 1.0)
    g = t[(t.snr_db == snr) & (t.alpha == a)].sort_values('delay')
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in METHODS:
        ax.errorbar(x, g[f'mean_{m}'], yerr=g[f'ci_{m}'], marker='o', ms=3, capsize=2, label=LABEL[m], color=COLOR[m])
    ax.set_xticks(x); ax.set_xticklabels(g.delay.astype(int))
    ax.set_yscale('log'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel('path-2 delay (samples); delay 10 = carrier null, 9 = sideband null')
    ax.set_ylabel('NMSE (mean, 95% CI)'); ax.set_title(f'Delay sweep at alpha = {a:g}, SNR = {snr:g} dB')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_FIG, 'fig4_delay_sweep.png'), dpi=140); plt.close(fig)


def fig_carrier_gain(t, snr):
    g = t[t.snr_db == snr]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in METHODS:
        axs[0].scatter(g.carrier_gain, g[f'mean_{m}'], s=16, label=LABEL[m], color=COLOR[m], alpha=0.7)
    axs[0].set_yscale('log'); axs[0].set_xlabel('carrier gain |H(fc)|'); axs[0].set_ylabel('mean NMSE')
    axs[0].set_title('Performance vs. how much carrier survives'); axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8)
    sig = g.sig_lms.values
    axs[1].scatter(g.carrier_gain[~sig], g.logratio_lms[~sig], s=16, c='gray', label='not significant')
    axs[1].scatter(g.carrier_gain[sig], g.logratio_lms[sig], s=22, c='k', label='significant')
    axs[1].axhline(0, color='r', lw=1)
    axs[1].set_xlabel('carrier gain |H(fc)|'); axs[1].set_ylabel('log10(NMSE_LMS / NMSE_sync)')
    axs[1].set_title('Adaptive vs sync (above 0: adaptive worse)'); axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8)
    fig.suptitle(f'SNR = {snr:g} dB'); fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, 'fig5_carrier_gain.png'), dpi=140); plt.close(fig)


# ---------------------------------------------------------------- main
def main(path):
    os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_TAB, exist_ok=True)
    df, files = load(path)
    integrity(df, files)
    t = point_table(df)
    t.to_csv(os.path.join(OUT_TAB, 'point_summary.csv'), index=False)
    headline(df, t)
    snr = nearest(t.snr_db, 15)
    fig_nmse_vs_snr(t); fig_heatmaps(t, snr); fig_ratio(t, snr); fig_null_sweep(t, snr); fig_carrier_gain(t, snr)
    print("=" * 64)
    print(f"Saved results/tables/point_summary.csv, headline_by_carrier_gain.csv and 5 figures in {OUT_FIG}/")
    print(f"(heatmaps / delay sweep / carrier-gain plots use SNR = {snr:g} dB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/raw/full')

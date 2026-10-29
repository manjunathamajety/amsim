"""
Monte Carlo sweep over the full pipeline.

Run from the project ROOT (the folder that contains src/):

    python -m src.sweep small            # quick check; prints an estimate of the full sweep's run time
    python -m src.sweep full             # run ALL blocks not yet finished
    python -m src.sweep full 1 2 3       # run only the blocks for delay = 1, 2, 3 (samples)
    python -m src.sweep status           # show which blocks are done / pending

A "block" = one delay value (all alphas x SNRs x runs) = one CSV in data/raw/full/.
Blocks can be run in any order and at any time; seeds depend only on the grid
position, so results are identical however you split the work. Stopping (Ctrl-C)
only loses the block in progress. To start over, delete the folder data/raw/full/.
"""
import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from .trial import run_single_trial, D_CARRIER
from .carrier_recovery import lms_carrier, rls_carrier

PRESETS = {
    'small': dict(
        delays=[3, 9, D_CARRIER], alphas=[0.5, 1.0], snrs=[10, 25],
        n_runs=5, outdir='data/raw/small'),
    'full': dict(
        delays=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20],        # samples; 9~sideband null, 10=carrier null
        alphas=[round(a, 1) for a in np.arange(0.1, 1.01, 0.1)],           # 10 values
        snrs=[round(s, 1) for s in np.linspace(0, 30, 12)],                # 12 values
        n_runs=200, outdir='data/raw/full'),
}
N_PROCS = max(1, cpu_count() - 1)       # leave one thread for the OS


def warm_up():
    z = np.zeros(100)
    lms_carrier(z, 20000.0, 400000.0, 2e-4)
    rls_carrier(z, 20000.0, 400000.0, 0.9998, 1.0)


def job(args):
    return run_single_trial(*args)


def main(mode, presets=PRESETS, only=None):
    cfg = presets[mode]
    os.makedirs(cfg['outdir'], exist_ok=True)

    # point id is fixed by position in the full grid -> seeds never depend on resume state
    points = list(itertools.product(cfg['delays'], cfg['alphas'], cfg['snrs']))
    n_runs = cfg['n_runs']
    total = len(points) * n_runs
    per_block = len(cfg['alphas']) * len(cfg['snrs']) * n_runs
    path_of = lambda d: os.path.join(cfg['outdir'], f"delay_{d:02d}.csv")
    done = [d for d in cfg['delays'] if os.path.exists(path_of(d))]
    pending = [d for d in cfg['delays'] if d not in done]
    print(f"[{mode}] {len(cfg['delays'])} blocks of {per_block} trials ({total} total)")
    print(f"  done:    {done}")
    print(f"  pending: {pending}")
    if only:
        bad = [d for d in only if d not in cfg['delays']]
        if bad:
            print(f"  ERROR: delays {bad} are not in this preset's delay list {cfg['delays']}")
            return
    run_list = [d for d in pending if (not only or d in only)]
    total = len(run_list) * per_block
    print(f"  running now: {run_list}  ({total} trials, {N_PROCS} processes)\n")

    warm_up()
    t0 = time.time()
    trials_done = 0
    with Pool(N_PROCS, initializer=warm_up) as pool:
        for d in run_list:
            path = path_of(d)
            jobs = [(dd, a, s, pid * 10_000 + r, 'multi')
                    for pid, (dd, a, s) in enumerate(points) if dd == d
                    for r in range(n_runs)]
            rows = pool.map(job, jobs, chunksize=max(1, len(jobs) // (N_PROCS * 8)))
            tmp = path + ".tmp"
            pd.DataFrame(rows).to_csv(tmp, index=False)
            os.replace(tmp, path)                      # atomic: no half-written files
            trials_done += len(jobs)
            el = time.time() - t0
            print(f"  delay {d:2d}: {len(jobs)} trials saved   "
                  f"({el/60:.1f} min elapsed, {trials_done/total*100:.0f}% of this session)")

    el = time.time() - t0
    print(f"\nFinished in {el/60:.1f} min. Files in {cfg['outdir']}/")

    if mode == 'small' and trials_done:
        ms = el / trials_done * 1000
        n_full = (len(presets['full']['delays']) * len(presets['full']['alphas'])
                  * len(presets['full']['snrs']) * presets['full']['n_runs'])
        print(f"Measured {ms:.1f} ms/trial wall-clock (includes pool start-up, so slightly pessimistic).")
        print(f"Estimated full sweep ({n_full} trials): about {n_full * ms / 1000 / 3600:.1f} hours.")


def status():
    cfg = PRESETS['full']
    done = [d for d in cfg['delays'] if os.path.exists(os.path.join(cfg['outdir'], f"delay_{d:02d}.csv"))]
    pending = [d for d in cfg['delays'] if d not in done]
    print(f"full sweep: {len(done)}/{len(cfg['delays'])} blocks done")
    print(f"  done:    {done}")
    print(f"  pending: {pending}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == 'status':
        status()
    elif args and args[0] in PRESETS:
        try:
            only = [int(a) for a in args[1:]] or None
            main(args[0], only=only)
        except KeyboardInterrupt:
            print("\nStopped. Finished blocks are saved; re-run to continue.")
    else:
        print("Usage: python -m src.sweep [small | full [delays...] | status]")
        sys.exit(1)

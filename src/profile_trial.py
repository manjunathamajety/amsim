"""
Where does the time go on THIS machine?   Run from the project root:

    python -m src.profile_trial          # default 56 trials for the scaling test
    python -m src.profile_trial 28       # fewer trials

Part 1: time of each stage of one trial (single process).
Part 2: throughput with 1 process vs N_PROCS processes -> is parallelism actually helping?
"""
import sys
import time
import numpy as np
from multiprocessing import Pool

from . import trial as T
from .carrier_recovery import lms_carrier, rls_carrier
from .signal_gen import generate_message, am_modulate
from .channel import build_channel_ir, apply_channel
from .demodulators import envelope_detector, synchronous_detector_locked
from .sweep import warm_up, job, N_PROCS

n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 56


def ms(f, n=10):
    f()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t0) / n * 1000


warm_up()
rng = np.random.default_rng(0)
m = generate_message(T.T)
s = am_modulate(m, T.T, T.FC, 0.7)
r = apply_channel(s, build_channel_ir([1.0, 0.6], [0, 5]), 15, rng)
fc, fs = float(T.FC), float(T.FS)

print(f"Signal length: {len(T.T)} samples\n")
print("PART 1: one trial, single process (ms)")
stages = {
    "generate + modulate":      lambda: am_modulate(generate_message(T.T), T.T, T.FC, 0.7),
    "channel + noise":          lambda: apply_channel(s, build_channel_ir([1.0, 0.6], [0, 5]), 15, rng),
    "envelope detector":        lambda: envelope_detector(r),
    "sync detector (locked)":   lambda: synchronous_detector_locked(r, T.T, T.FC, T.FS, T.W * 1.2),
    "LMS carrier (numba)":      lambda: lms_carrier(r, fc, fs, T.MU_LMS),
    "RLS carrier (numba)":      lambda: rls_carrier(r, fc, fs, T.LAM_RLS, T.DELTA),
    "2x low-pass filtfilt":     lambda: (T._lowpass(r), T._lowpass(r)),
    "WHOLE run_single_trial":   lambda: T.run_single_trial(5, 0.6, 15, 1, 'multi'),
}
for k, f in stages.items():
    print(f"  {k:26s} {ms(f):8.1f}")

print(f"\nPART 2: throughput on {n_trials} trials")
jobs = [(5, 0.6, 15, i, 'multi') for i in range(n_trials)]
res = {}
for procs in (1, N_PROCS):
    with Pool(procs, initializer=warm_up) as pool:
        pool.map(job, jobs[:procs * 2])                 # warm-up pass, not timed
        t0 = time.perf_counter()
        pool.map(job, jobs, chunksize=2)
        el = time.perf_counter() - t0
    res[procs] = el / n_trials * 1000
    print(f"  {procs} process(es): {res[procs]:7.1f} ms/trial wall-clock")
print(f"  speedup with {N_PROCS} processes: {res[1] / res[N_PROCS]:.1f}x   (expect ~4-5x on 4 cores / 8 threads)")
blk = 24000 * res[N_PROCS] / 1000 / 60
print(f"\nEstimate: one block (24000 trials) ~ {blk:.1f} min, full sweep (15 blocks) ~ {blk * 15 / 60:.1f} h")

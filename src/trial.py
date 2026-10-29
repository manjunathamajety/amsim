"""One full-pipeline trial: message -> AM -> multipath+AWGN -> 4 demodulators -> metrics."""
import numpy as np
from scipy.signal import butter, filtfilt

from .signal_gen import generate_message, am_modulate
from .channel import build_channel_ir, delay_for_null, apply_channel
from .demodulators import envelope_detector, synchronous_detector_locked
from .carrier_recovery import lms_carrier, rls_carrier
from .metrics import nmse, thd

# ---- fixed experiment settings ----
FS, FC, W, MOD_INDEX = 400_000, 20_000, 3_000, 0.7
DURATION = 0.2
SETTLE = 25_000          # samples ignored for ALL methods (adaptive filters need to settle)
MU_LMS = 2e-4            # LMS step size  (time constant ~ 1/mu samples)
LAM_RLS = 0.9998         # RLS forgetting (memory ~ 1/(1-lam) samples), matched to mu
DELTA = 1.0
THD_FREQ = 1_000         # tone used for THD runs

T = np.arange(0, DURATION, 1 / FS)
D_CARRIER, _ = delay_for_null(FC, FS)        # 10 samples -> null exactly at fc
_B, _A = butter(6, W * 1.2 / (FS / 2), btype='low')


def _lowpass(z):
    return filtfilt(_B, _A, z)


def run_single_trial(delay, alpha, snr_db, seed, message_kind='multi'):
    """delay: 2nd-path delay in samples; alpha: 2nd-path gain. message_kind: 'multi' or 'single'."""
    rng = np.random.default_rng(seed)
    if message_kind == 'single':
        m = generate_message(T, tones=((THD_FREQ, 1.0),))
    else:
        m = generate_message(T)
    s = am_modulate(m, T, FC, MOD_INDEX)
    h = build_channel_ir([1.0, alpha], [0, delay])
    r = apply_channel(s, h, snr_db, rng)

    outs = {
        'env':  envelope_detector(r),
        'sync': synchronous_detector_locked(r, T, FC, FS, W * 1.2),
        'lms':  _lowpass(r * lms_carrier(r, FC, FS, MU_LMS)),
        'rls':  _lowpass(r * rls_carrier(r, FC, FS, LAM_RLS, DELTA)),
    }

    res = {'delay': delay, 'alpha': alpha, 'snr_db': snr_db, 'seed': seed}
    for name, y in outs.items():
        res[f'nmse_{name}'] = nmse(y[SETTLE:], m[SETTLE:], trim=300)
        res[f'thd_{name}'] = thd(y[SETTLE:], FS, THD_FREQ, trim=300) if message_kind == 'single' else np.nan
    return res

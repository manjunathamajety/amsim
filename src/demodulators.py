import numpy as np
from scipy.signal import hilbert, butter, filtfilt

def envelope_detector(r):
    env = np.abs(hilbert(r))
    return env - np.mean(env)

def synchronous_detector(r, t, fc, fs, cutoff):
    mixed = r * np.cos(2*np.pi*fc*t)
    b, a = butter(6, cutoff/(fs/2), btype='low')
    return filtfilt(b, a, mixed)

def extract_carrier_reference(r, fc, fs, bw=200):
    """Narrowband bandpass around fc -> reference d(n) for the adaptive filters."""
    nyq = fs / 2
    b, a = butter(4, [(fc-bw)/nyq, (fc+bw)/nyq], btype='band')
    return filtfilt(b, a, r)


def estimate_carrier_phase(r, t, fc, fs, bw=200):
    """Phase of the received carrier (after the channel), from the narrowband component."""
    c = extract_carrier_reference(r, fc, fs, bw)
    return np.angle(np.sum(c * np.exp(-1j * 2*np.pi*fc*t)))

def synchronous_detector_locked(r, t, fc, fs, cutoff, bw=200):
    """Sync detector whose local oscillator is phase-aligned to the received carrier
    (stands in for a locked PLL). Falls apart when the carrier itself is nulled."""
    phi = estimate_carrier_phase(r, t, fc, fs, bw)
    mixed = r * np.cos(2*np.pi*fc*t + phi)
    b, a = butter(6, cutoff/(fs/2), btype='low')
    return filtfilt(b, a, mixed)

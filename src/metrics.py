import numpy as np

def nmse(est, ref, trim=200):
    """NMSE after removing DC and least-squares amplitude scaling; edges trimmed."""
    est = est[trim:-trim] - np.mean(est[trim:-trim])
    ref = ref[trim:-trim] - np.mean(ref[trim:-trim])
    denom = est @ est
    if denom == 0:
        return np.nan
    scale = (est @ ref) / denom
    return np.mean((scale*est - ref)**2) / np.mean(ref**2)

def thd(x, fs, f0, n_harm=5, trim=200):
    """THD of a demodulated SINGLE-TONE message at f0: harmonic power / fundamental."""
    x = x[trim:-trim] - np.mean(x[trim:-trim])
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    def peak(f):
        i = np.argmin(np.abs(freqs - f))
        return X[max(i-2, 0):i+3].max()
    fund = peak(f0)
    harm = np.sqrt(sum(peak(k*f0)**2 for k in range(2, n_harm+1)))
    return harm / fund

import numpy as np

def build_channel_ir(alphas, delays_samples):
    """Sparse FIR channel. Delays are in SAMPLES (integers), gains relative to path 1."""
    delays = np.asarray(delays_samples, dtype=int)
    h = np.zeros(delays.max() + 1)
    for a, d in zip(alphas, delays):
        h[d] += a
    return h

def delay_for_null(f_null, fs):
    """Integer-sample delay d for a 2-path (alpha=1) channel with a null near f_null.
    Null sits at fs/(2d). Returns (d, actual_null_freq) because rounding moves it."""
    d = int(round(fs / (2.0 * f_null)))
    return d, fs / (2.0 * d)

def apply_channel(s, h, snr_db, rng):
    """Multipath then AWGN. SNR is referenced to the TRANSMITTED signal power,
    so SNR and fade depth stay independent sweep axes."""
    r = np.convolve(s, h, mode='full')[:len(s)]
    noise_power = np.mean(s**2) / (10 ** (snr_db / 10))
    return r + np.sqrt(noise_power) * rng.standard_normal(len(r))

def freq_response(h, fs, nfft=16384):
    H = np.fft.rfft(h, n=nfft)
    return np.fft.rfftfreq(nfft, 1/fs), np.abs(H)

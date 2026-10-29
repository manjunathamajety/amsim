import numpy as np
from scipy.signal import hilbert, butter, filtfilt

def envelope_detector(r):
    env = np.abs(hilbert(r))
    return env - np.mean(env)

def synchronous_detector(r, t, fc, fs, cutoff):
    local_carrier = np.cos(2*np.pi*fc*t)
    mixed = r * local_carrier
    #low pass butterworth filter
    b, a = butter(N=6, Wn=cutoff/(fs/2), btype='low')
    return filtfilt(b, a, mixed)

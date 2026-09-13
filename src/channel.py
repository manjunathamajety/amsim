import numpy as np

def apply_awgn(signal, snr_db, rng):
    #simple signal for inital test
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * rng.standard_normal(len(signal))
    return signal + noise

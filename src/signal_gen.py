import numpy as np

def generate_message(t, tones=((500, 0.6), (1200, 0.3), (2500, 0.1))):
    """Multi-tone message. tones: (freq_hz, amplitude). Normalized to [-1, 1]."""
    m = np.zeros_like(t)
    for f, a in tones:
        m += a * np.sin(2*np.pi*f*t)
    return m / np.max(np.abs(m))

def am_modulate(message_n, t, fc, mu=0.7, Ac=1.0):
    """s(t) = Ac [1 + mu m(t)] cos(2 pi fc t)"""
    return Ac * (1 + mu*message_n) * np.cos(2*np.pi*fc*t)

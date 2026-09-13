import numpy as np

def generate_message(t, tones=((500, 0.6), (1200, 0.3), (2500, 0.1))):
    """Multi-tone message signal. tones: list of (freq_hz, amplitude)."""
    message = np.zeros_like(t)
    for freq, amp in tones:
        message += amp * np.sin(2*np.pi*freq*t)
    return message / np.max(np.abs(message))   # normalize to [-1, 1]

def am_modulate(message_n, t, fc, mu=0.7, Ac=1.0):
    """s(t) = Ac[1 + mu*m(t)]cos(2*pi*fc*t)"""
    return Ac * (1 + mu*message_n) * np.cos(2*np.pi*fc*t)

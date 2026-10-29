import numpy as np
import matplotlib.pyplot as plt

from .signal import generate_message, am_modulate
from .channel import apply_awgn
from .demodulators import envelope_detector, synchronous_detector

# ---- Parameters ----
fs = 500_000
fc = 20_000
duration = 0.01
W = 3_000
snr_db = 25

t = np.arange(0, duration, 1/fs)
rng = np.random.default_rng(42)

# ---- Generate + modulate ----
message_n = generate_message(t)
s = am_modulate(message_n, t, fc, mu=0.7)

# ---- Channel (just noise, no fading yet) ----
r = apply_awgn(s, snr_db, rng)

# ---- Demodulate ----
env_out = envelope_detector(r)
sync_out = synchronous_detector(r, t, fc, fs, cutoff=W*1.2)

# ---- Plot ----
fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
axs[0].plot(t[:2000], s[:2000]); axs[0].set_title("Transmitted AM signal")
axs[1].plot(t[:2000], r[:2000]); axs[1].set_title("Received signal (noisy)")
axs[2].plot(t[:2000], message_n[:2000], label="original message")
axs[2].plot(t[:2000], env_out[:2000], label="envelope detector")
axs[2].plot(t[:2000], sync_out[:2000], label="sync detector")
axs[2].legend(); axs[2].set_title("Demodulated vs original")
plt.tight_layout()
plt.show()

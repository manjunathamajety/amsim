"""Hand-picked cases through generator -> channel -> envelope/sync, plus null check.
Run:  python -m src.check_pipeline"""
import numpy as np
from .signal_gen import generate_message, am_modulate
from .channel import build_channel_ir, delay_for_null, apply_channel, freq_response
from .demodulators import envelope_detector, synchronous_detector
from .metrics import nmse

fs, fc, W, mu, snr = 400_000, 20_000, 3_000, 0.7, 25
t = np.arange(0, 0.05, 1/fs)
rng = np.random.default_rng(1)
m = generate_message(t)
s = am_modulate(m, t, fc, mu)

d_car, f_car = delay_for_null(fc, fs)
d_sb, f_sb = delay_for_null(fc + W, fs)
print(f"carrier null: delay={d_car} samples -> null at {f_car:.0f} Hz (target {fc})")
print(f"sideband null: delay={d_sb} samples -> null at {f_sb:.0f} Hz (target {fc+W})\n")

cases = {
    "clean (no 2nd path)":       ([1.0], [0]),
    "mild fade":                 ([1.0, 0.3], [0, 3]),
    "sideband null (alpha=1)":   ([1.0, 1.0], [0, d_sb]),
    "carrier null (alpha=1)":    ([1.0, 1.0], [0, d_car]),
}
for name, (al, dl) in cases.items():
    h = build_channel_ir(al, dl)
    f, H = freq_response(h, fs)
    Hc = H[np.argmin(np.abs(f - fc))]
    r = apply_channel(s, h, snr, rng)
    e = nmse(envelope_detector(r), m)
    y = nmse(synchronous_detector(r, t, fc, fs, W*1.2), m)
    print(f"{name:26s} |H(fc)|={Hc:5.2f}   NMSE env={e:8.4f}   sync={y:8.4f}")

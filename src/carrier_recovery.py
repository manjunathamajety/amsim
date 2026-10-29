"""
Adaptive carrier recovery (Widrow-style adaptive noise canceller with a sinusoidal reference).

Regressors are a locally generated cos/sin at fc. The 2 weights adapt so that
y(n) = w0*cos + w1*sin matches the CARRIER component of r(n); the message
sidebands act as noise that the slow-moving weights average out. y(n) is then
used as the local oscillator of a synchronous detector.

Call with all-positional args (Numba).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def lms_carrier(r, fc, fs, mu):
    N = len(r)
    y = np.zeros(N)
    w0 = 0.0
    w1 = 0.0
    om = 2.0 * np.pi * fc / fs
    for n in range(N):
        c = np.cos(om * n)
        s = np.sin(om * n)
        yn = w0 * c + w1 * s
        e = r[n] - yn
        y[n] = yn
        w0 += 2.0 * mu * e * c
        w1 += 2.0 * mu * e * s
    return y


@njit(cache=True)
def rls_carrier(r, fc, fs, lam, delta):
    N = len(r)
    y = np.zeros(N)
    w0 = 0.0
    w1 = 0.0
    P = np.eye(2) / delta
    om = 2.0 * np.pi * fc / fs
    for n in range(N):
        c = np.cos(om * n)
        s = np.sin(om * n)
        Px0 = P[0, 0] * c + P[0, 1] * s
        Px1 = P[1, 0] * c + P[1, 1] * s
        denom = lam + c * Px0 + s * Px1
        k0 = Px0 / denom
        k1 = Px1 / denom
        yn = w0 * c + w1 * s
        e = r[n] - yn
        y[n] = yn
        w0 += k0 * e
        w1 += k1 * e
        P[0, 0] = (P[0, 0] - k0 * Px0) / lam
        P[0, 1] = (P[0, 1] - k0 * Px1) / lam
        P[1, 0] = (P[1, 0] - k1 * Px0) / lam
        P[1, 1] = (P[1, 1] - k1 * Px1) / lam
    return y

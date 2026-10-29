"""
Numba-JIT versions of LMS and RLS.
Written with explicit loops (no slicing/@ on reversed views) so Numba
compiles them to tight machine code with no contiguity warnings.

NOTE: call with ALL positional args, e.g. rls_filter_njit(x, d, M, 0.999, 1.0)
"""
import numpy as np
from numba import njit


@njit(cache=True)
def lms_filter_njit(x, d, M, mu):
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(M - 1, N):
        acc = 0.0
        for i in range(M):
            acc += w[i] * x[n - i]          # xn[i] = x[n-i]
        y[n] = acc
        e[n] = d[n] - acc
        g = 2.0 * mu * e[n]
        for i in range(M):
            w[i] += g * x[n - i]
    return y, e, w


@njit(cache=True)
def rls_filter_njit(x, d, M, lam, delta):
    N = len(x)
    w = np.zeros(M)
    P = np.eye(M) / delta
    y = np.zeros(N)
    e = np.zeros(N)
    xn = np.zeros(M)
    Px = np.zeros(M)
    k = np.zeros(M)
    for n in range(M - 1, N):
        for i in range(M):
            xn[i] = x[n - i]
        # Px = P @ xn
        for i in range(M):
            s = 0.0
            for j in range(M):
                s += P[i, j] * xn[j]
            Px[i] = s
        # gain vector k = Px / (lam + xn.Px)
        denom = lam
        for i in range(M):
            denom += xn[i] * Px[i]
        for i in range(M):
            k[i] = Px[i] / denom
        # output / error (a priori, old weights)
        acc = 0.0
        for i in range(M):
            acc += w[i] * xn[i]
        y[n] = acc
        e[n] = d[n] - acc
        # weight update
        for i in range(M):
            w[i] += k[i] * e[n]
        # P update: (P - k Px^T) / lam
        for i in range(M):
            for j in range(M):
                P[i, j] = (P[i, j] - k[i] * Px[j]) / lam
    return y, e, w

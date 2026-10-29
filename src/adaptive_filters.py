import numpy as np

def lms_filter(x, d, M, mu):
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(M-1, N):
        xn = x[n-M+1:n+1][::-1]
        y[n] = w @ xn
        e[n] = d[n] - y[n]
        w = w + 2*mu*e[n]*xn
    return y, e, w

def rls_filter(x, d, M, lam=0.999, delta=1.0):
    N = len(x)
    w = np.zeros(M)
    P = np.eye(M) / delta
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(M-1, N):
        xn = x[n-M+1:n+1][::-1]
        Px = P @ xn
        k = Px / (lam + xn @ Px)
        y[n] = w @ xn
        e[n] = d[n] - y[n]
        w = w + k*e[n]
        P = (P - np.outer(k, Px)) / lam
    return y, e, w

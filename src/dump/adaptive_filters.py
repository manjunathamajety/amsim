import numpy as np

def lms_filter(x, d, M, mu):
    """
    x: input signal (received signal r(n))
    d: desired/reference signal (extracted carrier reference)
    M: filter length (number of taps)
    mu: step size
    Returns: y (filter output), e (error), w (final weights)
    """
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(M-1, N):
        xn = x[n-M+1:n+1][::-1]   # IMPORTANT: includes x[n] as the most recent tap
        y[n] = w @ xn
        e[n] = d[n] - y[n]
        w = w + 2*mu*e[n]*xn
    return y, e, w

def rls_filter(x, d, M, lam=0.999, delta=1.0):
    """
    lam: forgetting factor (see verification note below — too low causes divergence)
    delta: initial inverse-correlation scaling
    """
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

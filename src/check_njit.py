"""Confirms the JIT versions give the same answers as the plain versions.
Run:  python -m src.check_njit
"""
import numpy as np
from .adaptive_filters import lms_filter, rls_filter
from .adaptive_filters_njit import lms_filter_njit, rls_filter_njit

rng = np.random.default_rng(0)
x = rng.standard_normal(3000)
d = np.convolve(x, [1.0, 0.5, -0.3, 0.2], mode='full')[:3000] + 0.05*rng.standard_normal(3000)
M = 15

y1, e1, w1 = lms_filter(x, d, M, 0.001)
y2, e2, w2 = lms_filter_njit(x, d, M, 0.001)
print("LMS weights match:", np.allclose(w1, w2), "| outputs match:", np.allclose(y1, y2))

y1, e1, w1 = rls_filter(x, d, M, 0.999, 1.0)
y2, e2, w2 = rls_filter_njit(x, d, M, 0.999, 1.0)
print("RLS weights match:", np.allclose(w1, w2), "| outputs match:", np.allclose(y1, y2))

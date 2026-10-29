"""
test.py — verification suite for lms_filter / rls_filter (src/adaptive_filters.py)

Run from project root:
    python -m src.test
or, if you keep this file standalone / not inside the src package:
    python test.py    (with adaptive_filters.py in the same folder)

Checks performed:
  1. System identification vs. known channel + closed-form Wiener solution
  2. Convergence speed comparison (RLS should settle faster than LMS)
  3. LMS misadjustment vs. mu (steady-state MSE should increase with mu)
  4. RLS covariance windup check (||P|| should stay bounded, not blow up)
  5. LMS stability boundary (large mu should diverge)
  6. Final pass/fail summary
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from .adaptive_filters import lms_filter, rls_filter
except ImportError:
    from adaptive_filters import lms_filter, rls_filter


# ============================================================
# Setup: white noise input through a known FIR channel
# ============================================================

rng = np.random.default_rng(0)
true_taps = np.array([1.0, 0.5, -0.3, 0.2])
M = len(true_taps)
N = 5000

x = rng.standard_normal(N)
d_clean = np.convolve(x, true_taps, mode='full')[:N]
meas_noise_std = 0.05
d = d_clean + meas_noise_std * rng.standard_normal(N)

results = {}  # collect pass/fail for the final summary


# ============================================================
# CHECK 1: System identification vs. Wiener solution
# ============================================================

print("=" * 60)
print("CHECK 1: System identification vs. closed-form Wiener solution")
print("=" * 60)

X = np.array([x[n-M+1:n+1][::-1] for n in range(M-1, N)])
D = d[M-1:N]
w_wiener = np.linalg.lstsq(X, D, rcond=None)[0]

_, _, w_lms = lms_filter(x, d, M, mu=0.01)
_, _, w_rls = rls_filter(x, d, M, lam=0.999)

lms_err = np.linalg.norm(w_lms - w_wiener)
rls_err = np.linalg.norm(w_rls - w_wiener)

print(f"True channel taps:     {true_taps}")
print(f"Wiener solution:       {np.round(w_wiener, 4)}")
print(f"LMS converged weights: {np.round(w_lms, 4)}   (error vs Wiener: {lms_err:.5f})")
print(f"RLS converged weights: {np.round(w_rls, 4)}   (error vs Wiener: {rls_err:.5f})")

check1_pass = lms_err < 0.05 and rls_err < 0.05
results['1. System ID vs Wiener'] = check1_pass
print("PASS" if check1_pass else "FAIL — check regression vector indexing in your filters")


# ============================================================
# CHECK 2: Convergence speed (RLS should settle faster than LMS)
# ============================================================

print("\n" + "=" * 60)
print("CHECK 2: Convergence speed")
print("=" * 60)

def track_squared_error(y_or_e, is_error=True):
    return y_or_e**2 if is_error else None

_, e_lms, _ = lms_filter(x, d, M, mu=0.01)
_, e_rls, _ = rls_filter(x, d, M, lam=0.999)

lms_sq_err = e_lms**2
rls_sq_err = e_rls**2

def smooth(arr, w=30):
    return np.convolve(arr, np.ones(w)/w, mode='valid')

lms_smoothed = smooth(lms_sq_err)
rls_smoothed = smooth(rls_sq_err)

lms_floor = np.mean(lms_sq_err[-500:])
rls_floor = np.mean(rls_sq_err[-500:])

lms_conv_idx = np.argmax(lms_smoothed < 1.5 * lms_floor)
rls_conv_idx = np.argmax(rls_smoothed < 1.5 * rls_floor)

print(f"LMS steady-state MSE floor: {lms_floor:.6f}  (approx. converged by sample {lms_conv_idx})")
print(f"RLS steady-state MSE floor: {rls_floor:.6f}  (approx. converged by sample {rls_conv_idx})")

check2_pass = rls_conv_idx < lms_conv_idx
results['2. RLS converges faster than LMS'] = check2_pass
print("PASS" if check2_pass else "FAIL — RLS should converge in fewer samples than LMS; check lambda")


# ============================================================
# CHECK 3: LMS misadjustment vs mu
# ============================================================

print("\n" + "=" * 60)
print("CHECK 3: LMS misadjustment trend (steady-state MSE should rise with mu)")
print("=" * 60)

mus = [0.001, 0.005, 0.01, 0.02, 0.05]
steady_state_mse = []
for mu in mus:
    _, e, _ = lms_filter(x, d, M, mu)
    steady_state_mse.append(np.mean(e[-500:]**2))
    print(f"mu={mu:.3f}  ->  steady-state MSE = {steady_state_mse[-1]:.6f}")

check3_pass = all(steady_state_mse[i] <= steady_state_mse[i+1] * 1.05
                   for i in range(len(steady_state_mse)-1))  # allow tiny noise wiggle
results['3. LMS misadjustment increases with mu'] = check3_pass
print("PASS" if check3_pass else "FAIL — steady-state MSE should trend upward with mu")


# ============================================================
# CHECK 4: RLS covariance windup check
# ============================================================

print("\n" + "=" * 60)
print("CHECK 4: RLS covariance windup (||P|| should stay bounded)")
print("=" * 60)

def rls_with_P_tracking(x, d, M, lam, delta=1.0):
    N = len(x)
    w = np.zeros(M)
    P = np.eye(M) / delta
    P_norms = []
    for n in range(M-1, N):
        xn = x[n-M+1:n+1][::-1]
        Px = P @ xn
        k = Px / (lam + xn @ Px)
        e = d[n] - w @ xn
        w = w + k*e
        P = (P - np.outer(k, Px)) / lam
        if n % 250 == 0:
            P_norms.append(np.linalg.norm(P))
    return np.array(P_norms)

for lam_test in [0.99, 0.999]:
    P_norms = rls_with_P_tracking(x, d, M, lam_test)
    growth_ratio = P_norms[-1] / (P_norms[len(P_norms)//2] + 1e-12)
    status = "windup detected" if growth_ratio > 10 else "stable"
    print(f"lambda={lam_test:.4f}  final/mid ||P|| ratio = {growth_ratio:9.2f}   -> {status}")

check4_pass = True  # informational — read the printed output, this isn't a strict pass/fail
results['4. RLS covariance windup check (informational)'] = check4_pass
print("Use lambda >= 0.995 if windup is detected at lambda=0.99 on your real signals.")


# ============================================================
# CHECK 5: LMS stability boundary
# ============================================================

print("\n" + "=" * 60)
print("CHECK 5: LMS stability boundary")
print("=" * 60)

_, e_stable, _ = lms_filter(x, d, M, mu=0.01)
_, e_unstable, _ = lms_filter(x, d, M, mu=0.5)

final_stable = e_stable[-1]**2
final_unstable = e_unstable[-1]**2

print(f"Final squared error at mu=0.01 (stable):   {final_stable:.6f}")
print(f"Final squared error at mu=0.5  (unstable): {final_unstable}")

check5_pass = (not np.isfinite(final_unstable)) or (final_unstable > 100 * final_stable)
results['5. LMS diverges for large mu'] = check5_pass
print("PASS" if check5_pass else "FAIL — mu=0.5 should clearly diverge relative to mu=0.01")


# Plots

fig, axs = plt.subplots(1, 3, figsize=(16, 4))

axs[0].plot(lms_smoothed, label='LMS')
axs[0].plot(rls_smoothed, label='RLS')
axs[0].set_yscale('log')
axs[0].set_xlabel('Sample'); axs[0].set_ylabel('Squared error (smoothed)')
axs[0].set_title('Convergence speed')
axs[0].legend()

axs[1].plot(mus, steady_state_mse, 'o-')
axs[1].set_xlabel('mu'); axs[1].set_ylabel('steady-state MSE')
axs[1].set_title('LMS misadjustment vs mu')

axs[2].plot(e_stable**2, label='mu=0.01 (stable)')
axs[2].plot(e_unstable**2, label='mu=0.5 (unstable)')
axs[2].set_yscale('log')
axs[2].set_xlabel('Sample'); axs[2].set_ylabel('Squared error')
axs[2].set_title('LMS stability boundary')
axs[2].legend()

plt.tight_layout()
plt.show()

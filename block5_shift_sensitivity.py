"""
BLOCK 5 - Accelerating the power method and measuring the reliability of the axes
=================================================================================
(a) Shift:  C - sigma I  has eigenvalues lambda_i - sigma, same eigenvectors.
    For b1, the ratio becomes  max_{i>=2} |lambda_i - sigma| / |lambda_1 - sigma|.
    Optimal shift (real eigenvalues):  sigma* = (lambda_2 + lambda_D) / 2.

(b) Shifted inverse power method:  (C - sigma I)^{-1}  has eigenvalues 1/(lambda_i - sigma):
    the dominant one is the lambda_i closest to sigma.
    Ratio:  |lambda_j - sigma| / |lambda_nearest - sigma|  -> very small if sigma ~ lambda_j.
    At each step we solve (C - sigma I) w = v: factorise LU ONCE, then two
    substitutions (forward and backward) per iteration.

(c) Sensitivity: perturb the data with noise and measure how much each eigenvector rotates.
    Theoretical expectation: the rotation is large when the eigenvalue is close to its
    neighbours (small gap).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from synthetic_data import generate_squad
from block2_power_method import power_method


def inverse_power(C, sigma, v0, tol=1e-12, max_iter=1000):
    n = C.shape[0]
    fact = lu_factor(C - sigma * np.eye(n))        # LU with pivoting, only once
    v = v0 / np.linalg.norm(v0)
    lam_old = 0.0
    for k in range(1, max_iter + 1):
        w = lu_solve(fact, v)                      # solve (C - sigma I) w = v
        v = w / np.linalg.norm(w)
        lam = v @ C @ v                            # Rayleigh on the original matrix
        if abs(lam - lam_old) < tol * abs(lam):
            return lam, v, k
        lam_old = lam
    return lam, v, max_iter


def angle(u, v):
    return np.degrees(np.arccos(min(1.0, abs(u @ v))))


if __name__ == "__main__":
    _, X_raw, y, roles, names = generate_squad()
    N, D = X_raw.shape
    X = (X_raw - X_raw.mean(0)) / X_raw.std(0, ddof=1)
    C = X.T @ X / (N - 1)
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    rng = np.random.default_rng(3)
    v0 = rng.standard_normal(D)

    # ------------------------------------------------------------- (a) shift
    print("=== (a) Power method with shift, for b1 ===")
    sig_opt = (lam[1] + lam[-1]) / 2
    for sigma, label in [(0.0, "no shift"), (sig_opt, "optimal shift")]:
        ratio = max(abs(lam[1:] - sigma)) / abs(lam[0] - sigma)
        l, b, k, _ = power_method(C - sigma * np.eye(D), v0)
        print(f"  {label:13s} sigma = {sigma:6.3f}   theoretical ratio {ratio:.3f}   "
              f"iterations {k:3d}   lambda1 = {l + sigma:.10f}")

    # ------------------------------------------------------------- (b) inverse power
    print("\n=== (b) Shifted inverse power method (LU once, then substitutions) ===")
    print("  target        sigma    theoretical ratio  iterations   lambda found     error")
    for j, sigma in [(0, 3.8), (1, 3.0), (2, 1.0), (9, 0.0)]:
        dist = np.abs(lam - sigma)
        order = np.argsort(dist)
        ratio = dist[order[0]] / dist[order[1]]
        l, b, k = inverse_power(C, sigma, v0)
        print(f"  lambda_{j+1:<2d}   {sigma:6.2f}      {ratio:8.4f}          {k:4d}       "
              f"{l:.10f}   {abs(l - lam[j]):.1e}")
    print("  (lambda_10 with sigma = 0: 'pure' inverse power method, finds the smallest eigenvalue)")

    # ------------------------------------------------------------- concrete use: conditioning
    # C symmetric positive definite:  K_2(C) = lambda_max / lambda_min
    lam_max, _, _, _ = power_method(C, v0)
    lam_min, _, _ = inverse_power(C, 0.0, v0)
    s = np.linalg.svd(X, compute_uv=False)
    print("\n=== Conditioning (power method + inverse power method) ===")
    print(f"  K(C) = lambda_max / lambda_min = {lam_max:.4f} / {lam_min:.4f} = {lam_max / lam_min:.1f}")
    print(f"  K(X) from the SVD = s_max / s_min = {s[0] / s[-1]:.2f}   ->  K(X)^2 = {(s[0] / s[-1])**2:.1f} = K(C)")
    print("  forming C = X^T X squares the condition number: this is why the SVD of X is used")

    # ------------------------------------------------------------- (c) sensitivity
    print("\n=== (c) Sensitivity of the eigenvectors to noise in the data ===")
    level = 0.10                                     # noise: 10% of the standard deviation
    trials = 300
    ang = np.zeros((trials, 4))
    for p in range(trials):
        Xp = X + level * rng.standard_normal(X.shape)
        Xp = (Xp - Xp.mean(0)) / Xp.std(0, ddof=1)
        lp, Vp = np.linalg.eigh(Xp.T @ Xp / (N - 1))
        Vp = Vp[:, ::-1]
        ang[p] = [angle(V[:, j], Vp[:, j]) for j in range(4)]
    gap = [min(abs(lam[j] - lam[j - 1]) if j > 0 else np.inf, abs(lam[j] - lam[j + 1])) for j in range(4)]
    print("  axis   lambda   gap to neighbour    median rotation     90% rotation")
    for j in range(4):
        print(f"  b{j+1}    {lam[j]:6.3f}      {gap[j]:6.3f}            {np.median(ang[:, j]):6.2f} degrees"
              f"     {np.percentile(ang[:, j], 90):6.2f} degrees")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.boxplot([ang[:, j] for j in range(4)], widths=0.5, showfliers=False,
               medianprops=dict(color="#eb6834", lw=2),
               boxprops=dict(color="#2a78d6"), whiskerprops=dict(color="#2a78d6"),
               capprops=dict(color="#2a78d6"))
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([f"b{j+1}\ngap {gap[j]:.2f}" for j in range(4)])
    ax.set_ylabel("eigenvector rotation [degrees]")
    ax.set_title("10% noise on the data: how much each axis rotates", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color="#e4e3df", lw=0.6)
    fig.tight_layout(); fig.savefig("fig_sensitivity.png", dpi=160); plt.close(fig)
    print("\nfigure saved: fig_sensitivity.png")

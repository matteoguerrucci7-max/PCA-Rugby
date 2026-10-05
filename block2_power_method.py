"""
BLOCK 2 - Power method + deflation
==================================
Theory:
    v0 = sum_i c_i b_i   ->   C^k v0 = sum_i c_i lambda_i^k b_i
    the lambda_1 term dominates: error on the eigenvector ~ |lambda_2/lambda_1|^k
    (C symmetric: error on the Rayleigh eigenvalue ~ |lambda_2/lambda_1|^(2k))

Deflation (the orthogonality constraint of the Lagrangian, done in practice):
    C_1 = C - lambda_1 b_1 b_1^T   has eigenvalues 0, lambda_2, ..., lambda_D
    -> the power method on C_1 finds b_2, and so on.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from synthetic_data import generate_squad


def power_method(A, v0, tol=1e-12, max_iter=10_000):
    """Translation of the pseudocode, with a single product A*v per iteration."""
    v = v0 / np.linalg.norm(v0)
    lam_old = 0.0
    history = []                                  # (lambda, v) at each iteration
    for k in range(1, max_iter + 1):
        z = A @ v                                 # 1. new direction (only product)
        lam_new = v @ z                           # 3. Rayleigh with the old v (||v|| = 1)
        v = z / np.linalg.norm(z)                 # 2. normalisation
        history.append((lam_new, v.copy()))
        if abs(lam_new - lam_old) < tol * abs(lam_new):   # 4. relative stopping test
            return lam_new, v, k, history
        lam_old = lam_new
    print("WARNING: max_iter reached without convergence")
    return lam_new, v, max_iter, history


def principal_components(C, m, seed=0):
    """First m components with power method + deflation."""
    rng = np.random.default_rng(seed)
    A = C.copy()
    lam, B, iters, histories = [], [], [], []
    for j in range(m):
        v0 = rng.standard_normal(C.shape[0])      # random starting vector
        l, b, k, h = power_method(A, v0)
        lam.append(l); B.append(b); iters.append(k); histories.append(h)
        A = A - l * np.outer(b, b)                # deflation
    return np.array(lam), np.column_stack(B), iters, histories


if __name__ == "__main__":
    _, X_raw, y, roles, names = generate_squad()
    N = X_raw.shape[0]
    X = (X_raw - X_raw.mean(0)) / X_raw.std(0, ddof=1)
    C = X.T @ X / (N - 1)

    m = 3
    lam, B, iters, histories = principal_components(C, m)

    # reference: numpy's "exact" eigenvalues
    lam_ref, V_ref = np.linalg.eigh(C)
    idx = np.argsort(lam_ref)[::-1]
    lam_ref, V_ref = lam_ref[idx], V_ref[:, idx]

    print(" j   lambda power      lambda numpy     iterations   angle with numpy's b_j")
    for j in range(m):
        cosang = min(1.0, abs(B[:, j] @ V_ref[:, j]))
        ang = np.degrees(np.arccos(cosang))
        print(f" {j+1}   {lam[j]:12.8f}   {lam_ref[j]:12.8f}    {iters[j]:6d}        {ang:.1e} degrees")

    # ---- check of the convergence rate for b1
    ratio = lam_ref[1] / lam_ref[0]
    h = histories[0]
    err_v = np.array([np.sqrt(max(0.0, 1 - (v @ V_ref[:, 0])**2)) for _, v in h])  # sine of the angle
    err_l = np.array([abs(l - lam_ref[0]) for l, _ in h])
    k = np.arange(1, len(h) + 1)

    # slope measured on the central stretch (log of the error against k)
    sel = (err_v > 1e-13) & (k > 5)
    factor = np.exp(np.polyfit(k[sel], np.log(err_v[sel]), 1)[0])
    print(f"\ntheoretical |lambda2/lambda1| = {ratio:.4f}")
    print(f"measured reduction factor per iteration (eigenvector) = {factor:.4f}")

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.semilogy(k, err_v, color="#2a78d6", lw=2, label="eigenvector error (sine of the angle)")
    ax.semilogy(k, err_l, color="#eb6834", lw=2, label="eigenvalue error (Rayleigh)")
    kk = np.arange(1, len(h) + 1)
    ax.semilogy(kk, err_v[19] * ratio ** (kk - 20), color="#2a78d6", lw=1, ls="--",
                label=r"theory: $|\lambda_2/\lambda_1|^k$")
    ax.semilogy(kk, err_l[19] * ratio ** (2 * (kk - 20)), color="#eb6834", lw=1, ls="--",
                label=r"theory: $|\lambda_2/\lambda_1|^{2k}$")
    ax.set_ylim(1e-16, 10)
    ax.set_xlabel("iteration k"); ax.set_ylabel("error")
    ax.set_title("Power method on C: convergence to b1", loc="left")
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig("fig_power_convergence.png", dpi=160)
    print("figure saved: fig_power_convergence.png")

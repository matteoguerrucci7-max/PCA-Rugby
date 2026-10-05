"""
BLOCK 4 - Rank-k approximation (Eckart-Young) and missing data
==============================================================
Eckart-Young: among all rank-k matrices, the closest to X is
    X_k = U_k S_k V_k^T,   with   ||X - X_k||_2 = s_{k+1},
                                  ||X - X_k||_F = sqrt(s_{k+1}^2 + ... + s_r^2).

Missing data (the "messy" data): some players missed half the season.
Imputation with iterative SVD = fixed-point iteration:
    X^(0): holes filled with the column mean (0 after standardisation)
    repeat:  X_k = rank-k approximation of X^(t)
             X^(t+1) = observed data where available, X_k in the holes
    stop when the imputed values change by less than tol.
The idea: the statistics are correlated (few components explain almost everything), so
a player's observed statistics say a lot about the missing ones.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from synthetic_data import generate_squad, METRICS


def rank_k(X, k):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k], s


def impute_svd(X, mask, k, tol=1e-10, max_iter=1000):
    """X with zeros in the holes (mean, standardised data); mask = True where missing."""
    Xt = X.copy()
    changes = []
    for it in range(max_iter):
        Xk, _ = rank_k(Xt, k)
        new = Xk[mask]
        diff = np.linalg.norm(new - Xt[mask])
        changes.append(diff)
        Xt[mask] = new
        if diff < tol:
            break
    return Xt, np.array(changes)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e4e3df", lw=0.6)


if __name__ == "__main__":
    X_nan, X_full, y, roles, names = generate_squad()
    N, D = X_full.shape

    # ---------------------------------------------------------- 1. Eckart-Young on the complete data
    X = (X_full - X_full.mean(0)) / X_full.std(0, ddof=1)
    print("=== 1. Eckart-Young ===")
    print("  k   ||X-X_k||_2   s_{k+1}    ||X-X_k||_F   sqrt(sum s^2)    relative error F")
    _, s = rank_k(X, 1)
    errF = []
    for k in range(1, D):
        Xk, _ = rank_k(X, k)
        e2 = np.linalg.norm(X - Xk, 2)
        eF = np.linalg.norm(X - Xk, "fro")
        errF.append(eF / np.linalg.norm(X, "fro"))
        print(f"  {k:2d}   {e2:10.6f}  {s[k]:10.6f}   {eF:10.6f}   {np.sqrt(np.sum(s[k:]**2)):12.6f}"
              f"      {errF[-1]:6.1%}")

    lam = s**2 / (N - 1)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axs[0].bar(np.arange(1, D + 1), lam, color="#2a78d6", width=0.6)
    axs[0].axhline(1.0, color="#52514e", lw=0.8, ls=":")
    axs[0].text(D + 0.4, 1.02, "mean = 1", ha="right", va="bottom", fontsize=8.5, color="#52514e")
    axs[0].set_xticks(range(1, D + 1))
    axs[0].set_xlabel("component j"); axs[0].set_ylabel("eigenvalue λ_j (variance)")
    axs[0].set_title("Eigenvalues (scree plot)", loc="left")
    axs[1].plot(range(1, D), np.array(errF) * 100, "o-", color="#eb6834", ms=5)
    axs[1].set_xticks(range(1, D))
    axs[1].set_xlabel("rank k"); axs[1].set_ylabel("relative error ||X−X_k||_F  [%]")
    axs[1].set_title("Error of the rank-k approximation", loc="left")
    for a in axs:
        style(a)
    fig.tight_layout(); fig.savefig("fig_eckart_young.png", dpi=160); plt.close(fig)

    # ---------------------------------------------------------- 2. imputation of the missing data
    mask = np.isnan(X_nan)
    rows = np.where(mask.any(1))[0]
    print("\n=== 2. Missing data ===")
    for i in rows:
        print(f"  {names[i]}: missing {', '.join(METRICS[j] for j in np.where(mask[i])[0])}")

    mean = np.nanmean(X_nan, 0)
    dev = np.nanstd(X_nan, 0, ddof=1)
    Xs = (X_nan - mean) / dev
    Xs[mask] = 0.0                                  # = replacement with the mean
    true = ((X_full - mean) / dev)[mask]            # true values, same scale

    rmse_mean = np.sqrt(np.mean((0.0 - true) ** 2))
    print(f"\n  error (RMSE, in standard deviations) on the missing values:")
    print(f"  replacement with the mean: {rmse_mean:.3f}")
    results = {}
    for k in range(1, 7):
        Xi, ch = impute_svd(Xs, mask, k)
        rmse = np.sqrt(np.mean((Xi[mask] - true) ** 2))
        results[k] = (rmse, ch)
        print(f"  iterative SVD, k = {k}:    {rmse:.3f}   ({len(ch)} iterations)")

    # convergence of the fixed point for k = 3
    ch3 = results[3][1]
    fact = np.exp(np.polyfit(np.arange(10, 40), np.log(ch3[10:40]), 1)[0]) if len(ch3) > 45 else np.nan
    print(f"\n  k = 3: reduction factor of the change per iteration = {fact:.3f}  (linear convergence)")

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ks = list(results)
    axs[0].axhline(rmse_mean, color="#52514e", lw=1.2, ls="--")
    axs[0].text(6.2, rmse_mean, "mean", va="center", fontsize=8.5, color="#52514e")
    axs[0].plot(ks, [results[k][0] for k in ks], "o-", color="#2a78d6", ms=6)
    axs[0].set_xlabel("rank k used for imputation"); axs[0].set_ylabel("RMSE on the missing values")
    axs[0].set_title("Imputation: iterative SVD against the mean", loc="left")
    axs[1].semilogy(np.arange(1, len(ch3) + 1), ch3, color="#2a78d6", lw=2)
    axs[1].set_xlabel("iteration"); axs[1].set_ylabel("change of the imputed values")
    axs[1].set_title("Fixed point (k = 3): linear convergence", loc="left")
    for a in axs:
        style(a)
    fig.tight_layout(); fig.savefig("fig_imputation.png", dpi=160); plt.close(fig)
    print("\nfigures saved: fig_eckart_young.png, fig_imputation.png")

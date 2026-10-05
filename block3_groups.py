"""
BLOCK 3 - Balanced training groups and work directions
=======================================================
Theory:
    score of player n in the latent space:  z_n = B^T x_n   (k components)
    team mean = 0 (centred data)

    SS_tot     = sum_n ||z_n||^2                          (fixed)
    SS_between = sum_g n_g ||zbar_g||^2                   (groups differ from each other)
    SS_within  = sum_g sum_{n in g} ||z_n - zbar_g||^2    (variety inside the group)
    SS_tot = SS_between + SS_within

    Goal: min SS_between  <=>  max SS_within  (strong and weak in the same group).
    The axes must not be reweighted: the score along b_j already has variance lambda_j.

Algorithm (swap heuristic):
    start from random groups; try every swap of two players from different
    groups; apply the swap that lowers SS_between the most; repeat until
    no swap improves (local minimum). Repeat from several random starts.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from synthetic_data import generate_squad, METRICS
from block2_power_method import principal_components

K = 3            # components kept (about 80% of the variance)
G = 5            # number of groups, of 5 players each
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


# ------------------------------------------------------------------ decomposition
def decomposition(Z, groups):
    """Return SS_tot, SS_between, SS_within (Z centred: team mean = 0)."""
    ss_tot = np.sum(Z**2)
    ss_between, ss_within = 0.0, 0.0
    for g in np.unique(groups):
        Zg = Z[groups == g]
        mean_g = Zg.mean(axis=0)
        ss_between += len(Zg) * mean_g @ mean_g
        ss_within += np.sum((Zg - mean_g) ** 2)
    return ss_tot, ss_between, ss_within


def ss_between(Z, groups):
    return decomposition(Z, groups)[1]


# ------------------------------------------------------------------ optimisation
def balance(Z, G, n_starts=30, seed=0):
    rng = np.random.default_rng(seed)
    N = len(Z)
    best, best_val = None, np.inf
    for _ in range(n_starts):
        groups = rng.permutation(np.arange(N) % G)          # random groups, 5 per group
        val = ss_between(Z, groups)
        while True:
            best_delta, best_swap = 0.0, None
            for i in range(N):
                for j in range(i + 1, N):
                    if groups[i] == groups[j]:
                        continue
                    groups[i], groups[j] = groups[j], groups[i]      # try the swap
                    delta = ss_between(Z, groups) - val
                    groups[i], groups[j] = groups[j], groups[i]      # undo it
                    if delta < best_delta - 1e-12:
                        best_delta, best_swap = delta, (i, j)
            if best_swap is None:                                     # local minimum
                break
            i, j = best_swap
            groups[i], groups[j] = groups[j], groups[i]
            val += best_delta
        if val < best_val:
            best, best_val = groups.copy(), val
    return best, best_val


# ------------------------------------------------------------------ work directions
def work_directions(X, Z, lam, B, roles, names, groups, axes=(0, 1), threshold=0.25):
    """
    The axes describe STYLES, not levels: each player is compared with the mean of
    their role.
    1) Axis: on b_j the "typical direction of the role" is s = sign(role mean). The axis
       is used only if the role clearly sits on one side (|mean| >= 0.5 sqrt(lambda_j)).
           deficit_j = s * (z_j - role_mean_j) / sqrt(lambda_j)
       Pick the axis with the most negative deficit; below -threshold it is room for work.
    2) Statistics: among those with a large weight in direction s of b_j, keep only those
       the role really uses (role mean above the team mean) and in which the player
       is below the mean of their role.
    3) Mentor: the teammate in the same group who is strongest on those statistics.
    """
    N = len(Z)
    role_idx = {r: [i for i in range(N) if roles[i] == r] for r in set(roles)}
    mz = {r: Z[ix].mean(axis=0) for r, ix in role_idx.items()}      # role mean, latent space
    mx = {r: X[ix].mean(axis=0) for r, ix in role_idx.items()}      # role mean, statistics
    rows = []
    for n in range(N):
        r = roles[n]
        best = None
        for j in axes:
            if abs(mz[r][j]) < 0.5 * np.sqrt(lam[j]):
                continue                        # the role is not characterised by this axis
            s = np.sign(mz[r][j])
            deficit = s * (Z[n, j] - mz[r][j]) / np.sqrt(lam[j])
            if best is None or deficit < best[1]:
                best = (j, deficit, s)
        if best is None or best[1] > -threshold:
            rows.append((names[n], "-", "in line with or above the role mean", "-"))
            continue
        j, deficit, s = best
        order = np.argsort(s * B[:, j])[::-1]
        stat = [t for t in order[:6] if mx[r][t] > 0 and X[n, t] < mx[r][t]][:3]
        if not stat:
            stat = [order[0]]
        mates = [i for i in range(N) if groups[i] == groups[n] and i != n]
        mentor = max(mates, key=lambda i: X[i, stat].mean())
        rows.append((names[n], f"b{j+1} ({deficit:+.2f})",
                     ", ".join(METRICS[t] for t in stat), names[mentor]))
    return rows


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    _, X_raw, y, roles, names = generate_squad()
    N = X_raw.shape[0]
    X = (X_raw - X_raw.mean(0)) / X_raw.std(0, ddof=1)
    C = X.T @ X / (N - 1)
    lam, B, _, _ = principal_components(C, K)

    # sign convention (b and -b are equivalent): the largest weight in absolute value is positive
    for j in range(K):
        if B[np.argmax(np.abs(B[:, j])), j] < 0:
            B[:, j] *= -1
    Z = X @ B
    print("axes: b1 = contact (+) / wide game (-)")
    print("      b2 = ball carrier (+) / low carriers: playmakers and props (-)")
    print(f"explained variance with K={K}: {lam.sum() / np.trace(C):.1%}\n")

    # --- optimisation
    groups, val = balance(Z, G)
    ss_tot, ss_b, ss_w = decomposition(Z, groups)
    print("=== Variance decomposition (optimised groups) ===")
    print(f"  SS_tot = {ss_tot:.4f}   SS_between = {ss_b:.4f}   SS_within = {ss_w:.4f}")
    print(f"  check SS_between + SS_within - SS_tot = {ss_b + ss_w - ss_tot:.1e}")
    print(f"  share between groups: {ss_b / ss_tot:.2%}")

    # --- comparison with random groups
    rng = np.random.default_rng(1)
    random_ss = np.array([ss_between(Z, rng.permutation(np.arange(N) % G)) for _ in range(5000)])
    print("\n=== Comparison with 5000 random splits ===")
    print(f"  random SS_between: mean {random_ss.mean():.3f}, minimum {random_ss.min():.3f}")
    print(f"  optimised SS_between: {val:.4f}  ({val / random_ss.mean():.1%} of the random mean)")
    print(f"  random splits better than the optimised one: {(random_ss <= val).mean():.2%}")

    # --- group composition
    print("\n=== Groups ===")
    for g in range(G):
        idx = np.where(groups == g)[0]
        mean = Z[idx].mean(axis=0)
        print(f"  Group {g+1}  mean (b1,b2,b3) = ({mean[0]:+.2f}, {mean[1]:+.2f}, {mean[2]:+.2f})")
        print("     " + ", ".join(names[i] for i in idx))

    # --- work directions
    print("\n=== Work directions (comparison with the player's own role) ===")
    print(f"  {'player':18s} {'axis':12s} {'to train':52s} mentor")
    for name, axis, top, mentor in work_directions(X, Z, lam, B, roles, names, groups):
        print(f"  {name:18s} {axis:12s} {top:52s} {mentor}")

    # --- figure 1: players on the b1-b2 plane, coloured by group
    abbr = {"Prop": "PR", "Hooker": "HK", "Lock": "LK", "Back row": "BR",
            "Scrum-half": "SH", "Fly-half": "FH", "Centre": "CE", "Wing": "WG", "Fullback": "FB"}
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for g in range(G):
        idx = np.where(groups == g)[0]
        ax.scatter(Z[idx, 0], Z[idx, 1], s=70, color=COL[g], edgecolor="white", lw=1.5,
                   label=f"Group {g+1}", zorder=3)
        m = Z[idx, :2].mean(axis=0)
        ax.scatter(*m, marker="X", s=110, color=COL[g], edgecolor="white", lw=1.2, zorder=4)
    for n in range(N):
        ax.annotate(abbr[roles[n]], (Z[n, 0], Z[n, 1]), xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, color="#52514e")
    ax.axhline(0, color="#c3c2b7", lw=0.8); ax.axvline(0, color="#c3c2b7", lw=0.8)
    ax.set_xlabel("b1: wide game  ←→  contact")
    ax.set_ylabel("b2: low carrier (playmakers, props)  ←→  ball carrier")
    ax.set_title("Players in the style space; X = group mean", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=5, bbox_to_anchor=(0, -0.12))
    fig.tight_layout(); fig.savefig("fig_groups.png", dpi=160); plt.close(fig)

    # --- figure 2: optimised against random
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.hist(random_ss, bins=50, color="#86b6ef", edgecolor="white", lw=0.6)
    ax.axvline(val, color="#eb6834", lw=2)
    ax.annotate(f"optimised groups\nSS_between = {val:.2f}", xy=(val, ax.get_ylim()[1] * 0.8),
                xytext=(12, 0), textcoords="offset points", fontsize=9, color="#0b0b0b")
    ax.set_xlabel("SS_between (imbalance between groups)"); ax.set_ylabel("number of random splits")
    ax.set_title("5000 random splits against the optimised one", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig("fig_random_comparison.png", dpi=160); plt.close(fig)
    print("\nfigures saved: fig_groups.png, fig_random_comparison.png")

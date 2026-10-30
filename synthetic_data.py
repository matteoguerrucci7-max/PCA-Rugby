"""
Synthetic but realistic dataset for a rugby union (XV) squad.

Each row is a player, each column a season statistic normalised to 80 minutes.
The mean profile of each role is built by hand; every player is that profile
plus multiplicative and additive Gaussian noise (clipped at 0).
Some roles are "polarised" (props, locks, wings, scrum-halves) and some are
"hybrid" (back row, centres), sitting between forwards and backs.
"""
import numpy as np

METRICS = [
    "metres_gained",
    "dominant_tackles",
    "attacking_rucks",
    "passes",
    "offloads",
    "defenders_beaten",
    "lineout_jumps",
    "kicks_in_play",
    "tackles_made",
    "carries",
]

# Mean profiles per 80 minutes (same order as METRICS)
ROLE_PROFILES = {
    #               metres domT  ruck  pass  off  defB  lineo kicks tack  carr
    "Prop":         [18,   1.0, 15.0,  3,   0.3, 0.6,  0.2,  0.0, 10,   7],
    "Hooker":       [24,   1.2, 13.0,  5,   0.5, 1.0,  0.1,  0.0, 12,   8],
    "Lock":         [22,   1.4, 14.0,  4,   0.4, 0.8,  6.5,  0.0, 13,   8],
    "Back row":     [44,   1.8,  9.0,  8,   1.1, 2.4,  1.5,  0.1, 12,  11],
    "Scrum-half":   [28,   0.4,  2.5, 65,   0.3, 1.5,  0.0,  4.0,  6,   4],
    "Fly-half":     [34,   0.5,  2.5, 28,   0.5, 2.0,  0.0, 12.0,  7,   6],
    "Centre":       [50,   1.4,  6.0, 10,   1.2, 3.2,  0.0,  0.8, 11,  11],
    "Wing":         [72,   0.3,  2.5,  5,   0.8, 4.6,  0.0,  1.5,  5,   9],
    "Fullback":     [66,   0.3,  2.5,  7,   0.7, 4.0,  0.0,  6.0,  5,   9],
}

# Squad composition: 14 forwards + 11 backs = 25
SQUAD = [
    ("Prop", 4), ("Hooker", 2), ("Lock", 3), ("Back row", 5),
    ("Scrum-half", 2), ("Fly-half", 2), ("Centre", 3), ("Wing", 3), ("Fullback", 1),
]

FORWARDS = {"Prop", "Hooker", "Lock", "Back row"}
HYBRID_ROLES = {"Back row", "Centre"}


def generate_squad(seed=7, rel_noise=0.18, missing_players=2, missing_frac=0.4):
    """
    Generate the synthetic squad.

    Parameters
    ----------
    seed : int
        Seed of the random generator (reproducibility).
    rel_noise : float
        Relative standard deviation of the noise with respect to the mean profile.
    missing_players : int
        Number of players who "missed half the season": for them a fraction
        of the statistics is set to NaN.
    missing_frac : float
        Fraction of missing statistics for each of those players.

    Returns
    -------
    X : ndarray (25, p)      data with NaNs
    X_full : ndarray (25, p) complete data (ground truth for the imputation)
    y : ndarray (25,)        +1 forwards, -1 backs
    roles : list[str]        official role of each player
    names : list[str]        player identifier
    """
    rng = np.random.default_rng(seed)
    roles, rows = [], []
    for role, count in SQUAD:
        mu = np.asarray(ROLE_PROFILES[role], dtype=float)
        for _ in range(count):
            # multiplicative noise + small additive component, clipped at >= 0
            sample = mu * (1 + rel_noise * rng.standard_normal(mu.size))
            sample += 0.15 * rng.standard_normal(mu.size)
            rows.append(np.clip(sample, 0.0, None))
            roles.append(role)

    X_full = np.vstack(rows)
    y = np.array([+1 if r in FORWARDS else -1 for r in roles], dtype=float)

    counters = {}
    names = []
    for r in roles:
        counters[r] = counters.get(r, 0) + 1
        names.append(f"{r} {counters[r]}")

    # Players who missed half the season: NaN on some statistics
    X = X_full.copy()
    p = X.shape[1]
    n_miss = max(1, int(round(missing_frac * p)))
    for i in rng.choice(len(roles), size=missing_players, replace=False):
        cols = rng.choice(p, size=n_miss, replace=False)
        X[i, cols] = np.nan

    return X, X_full, y, roles, names

"""
BLOCCO 5 - Accelerare il metodo delle potenze e misurare l'affidabilita' degli assi
===================================================================================
(a) Shift:  C - sigma I  ha autovalori lambda_i - sigma, stessi autovettori.
    Per b1, il rapporto diventa  max_{i>=2} |lambda_i - sigma| / |lambda_1 - sigma|.
    Shift ottimo (autovalori reali):  sigma* = (lambda_2 + lambda_D) / 2.

(b) Potenze inverse con shift:  (C - sigma I)^{-1}  ha autovalori 1/(lambda_i - sigma):
    il dominante e' quello con lambda_i piu' vicino a sigma.
    Rapporto:  |lambda_j - sigma| / |lambda_vicino - sigma|  -> piccolissimo se sigma ~ lambda_j.
    A ogni passo si risolve (C - sigma I) w = v: fattorizzo LU UNA volta, poi due
    sostituzioni (avanti e indietro) per iterazione.

(c) Sensibilita': perturbo i dati con rumore e misuro di quanto ruota ogni autovettore.
    Attesa teorica: la rotazione e' grande quando l'autovalore e' vicino ai suoi vicini
    (gap piccolo).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from synthetic_data import generate_squad
from blocco2_potenze import metodo_potenze


def potenze_inverse(C, sigma, v0, tol=1e-12, max_iter=1000):
    n = C.shape[0]
    fatt = lu_factor(C - sigma * np.eye(n))        # LU con pivoting, una volta sola
    v = v0 / np.linalg.norm(v0)
    lam_old = 0.0
    for k in range(1, max_iter + 1):
        w = lu_solve(fatt, v)                      # risolvo (C - sigma I) w = v
        v = w / np.linalg.norm(w)
        lam = v @ C @ v                            # Rayleigh sulla matrice originale
        if abs(lam - lam_old) < tol * abs(lam):
            return lam, v, k
        lam_old = lam
    return lam, v, max_iter


def angolo(u, v):
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
    print("=== (a) Metodo delle potenze con shift, per b1 ===")
    sig_opt = (lam[1] + lam[-1]) / 2
    for sigma, nome in [(0.0, "senza shift"), (sig_opt, "shift ottimo")]:
        rapp = max(abs(lam[1:] - sigma)) / abs(lam[0] - sigma)
        l, b, k, _ = metodo_potenze(C - sigma * np.eye(D), v0)
        print(f"  {nome:13s} sigma = {sigma:6.3f}   rapporto teorico {rapp:.3f}   "
              f"iterazioni {k:3d}   lambda1 = {l + sigma:.10f}")

    # ------------------------------------------------------------- (b) potenze inverse
    print("\n=== (b) Potenze inverse con shift (LU una volta, poi sostituzioni) ===")
    print("  cerco         sigma    rapporto teorico   iterazioni   lambda trovato   errore")
    for j, sigma in [(0, 3.8), (1, 3.0), (2, 1.0), (9, 0.0)]:
        dist = np.abs(lam - sigma)
        ordine = np.argsort(dist)
        rapp = dist[ordine[0]] / dist[ordine[1]]
        l, b, k = potenze_inverse(C, sigma, v0)
        print(f"  lambda_{j+1:<2d}   {sigma:6.2f}      {rapp:8.4f}          {k:4d}       "
              f"{l:.10f}   {abs(l - lam[j]):.1e}")
    print("  (lambda_10 con sigma = 0: potenze inverse 'pure', trovano l'autovalore piu' piccolo)")

    # ------------------------------------------------------------- uso concreto: condizionamento
    # C simmetrica definita positiva:  K_2(C) = lambda_max / lambda_min
    lam_max, _, _, _ = metodo_potenze(C, v0)
    lam_min, _, _ = potenze_inverse(C, 0.0, v0)
    s = np.linalg.svd(X, compute_uv=False)
    print("\n=== Condizionamento (potenze + potenze inverse) ===")
    print(f"  K(C) = lambda_max / lambda_min = {lam_max:.4f} / {lam_min:.4f} = {lam_max / lam_min:.1f}")
    print(f"  K(X) dalla SVD = s_max / s_min = {s[0] / s[-1]:.2f}   ->  K(X)^2 = {(s[0] / s[-1])**2:.1f} = K(C)")
    print("  formare C = X^T X eleva al quadrato il condizionamento: per questo si usa la SVD di X")

    # ------------------------------------------------------------- (c) sensibilita'
    print("\n=== (c) Sensibilita' degli autovettori al rumore nei dati ===")
    livello = 0.10                                   # rumore: 10% della deviazione standard
    prove = 300
    ang = np.zeros((prove, 4))
    for p in range(prove):
        Xp = X + livello * rng.standard_normal(X.shape)
        Xp = (Xp - Xp.mean(0)) / Xp.std(0, ddof=1)
        lp, Vp = np.linalg.eigh(Xp.T @ Xp / (N - 1))
        Vp = Vp[:, ::-1]
        ang[p] = [angolo(V[:, j], Vp[:, j]) for j in range(4)]
    gap = [min(abs(lam[j] - lam[j - 1]) if j > 0 else np.inf, abs(lam[j] - lam[j + 1])) for j in range(4)]
    print("  asse   lambda   gap con il vicino   rotazione mediana   rotazione 90%")
    for j in range(4):
        print(f"  b{j+1}    {lam[j]:6.3f}      {gap[j]:6.3f}            {np.median(ang[:, j]):6.2f} gradi"
              f"      {np.percentile(ang[:, j], 90):6.2f} gradi")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.boxplot([ang[:, j] for j in range(4)], widths=0.5, showfliers=False,
               medianprops=dict(color="#eb6834", lw=2),
               boxprops=dict(color="#2a78d6"), whiskerprops=dict(color="#2a78d6"),
               capprops=dict(color="#2a78d6"))
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([f"b{j+1}\ngap {gap[j]:.2f}" for j in range(4)])
    ax.set_ylabel("rotazione dell'autovettore [gradi]")
    ax.set_title("Rumore del 10% sui dati: quanto ruota ogni asse", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color="#e4e3df", lw=0.6)
    fig.tight_layout(); fig.savefig("fig_sensibilita.png", dpi=160); plt.close(fig)
    print("\ngrafico salvato: fig_sensibilita.png")

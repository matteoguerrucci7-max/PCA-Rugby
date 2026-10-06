"""
BLOCCO 4 - Approssimazione di rango k (Eckart-Young) e dati mancanti
=====================================================================
Eckart-Young: tra tutte le matrici di rango k, la piu' vicina a X e'
    X_k = U_k S_k V_k^T,   con   ||X - X_k||_2 = s_{k+1},
                                 ||X - X_k||_F = sqrt(s_{k+1}^2 + ... + s_r^2).

Dati mancanti (i dati "disordinati"): alcuni giocatori hanno saltato meta' stagione.
Imputazione con SVD iterativa = iterazione di punto fisso:
    X^(0): buchi riempiti con la media della colonna (0 dopo la standardizzazione)
    ripeti:  X_k = approssimazione di rango k di X^(t)
             X^(t+1) = dati osservati dove ci sono, X_k nei buchi
    stop quando i valori imputati cambiano meno di tol.
L'idea: le statistiche sono correlate (poche componenti spiegano quasi tutto), quindi
le statistiche osservate di un giocatore dicono molto su quelle mancanti.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from synthetic_data import generate_squad, METRICS


def rango_k(X, k):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k], s


def imputa_svd(X, mask, k, tol=1e-10, max_iter=1000):
    """X con zeri nei buchi (media, dati standardizzati); mask = True dove manca."""
    Xt = X.copy()
    variazioni = []
    for it in range(max_iter):
        Xk, _ = rango_k(Xt, k)
        nuovi = Xk[mask]
        diff = np.linalg.norm(nuovi - Xt[mask])
        variazioni.append(diff)
        Xt[mask] = nuovi
        if diff < tol:
            break
    return Xt, np.array(variazioni)


def stile(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e4e3df", lw=0.6)


if __name__ == "__main__":
    X_nan, X_full, y, roles, names = generate_squad()
    N, D = X_full.shape

    # ---------------------------------------------------------- 1. Eckart-Young sui dati completi
    X = (X_full - X_full.mean(0)) / X_full.std(0, ddof=1)
    print("=== 1. Eckart-Young ===")
    print("  k   ||X-X_k||_2   s_{k+1}    ||X-X_k||_F   sqrt(somma s^2)   errore relativo F")
    _, s = rango_k(X, 1)
    errF = []
    for k in range(1, D):
        Xk, _ = rango_k(X, k)
        e2 = np.linalg.norm(X - Xk, 2)
        eF = np.linalg.norm(X - Xk, "fro")
        errF.append(eF / np.linalg.norm(X, "fro"))
        print(f"  {k:2d}   {e2:10.6f}  {s[k]:10.6f}   {eF:10.6f}   {np.sqrt(np.sum(s[k:]**2)):12.6f}"
              f"      {errF[-1]:6.1%}")

    lam = s**2 / (N - 1)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axs[0].bar(np.arange(1, D + 1), lam, color="#2a78d6", width=0.6)
    axs[0].axhline(1.0, color="#52514e", lw=0.8, ls=":")
    axs[0].text(D + 0.4, 1.02, "media = 1", ha="right", va="bottom", fontsize=8.5, color="#52514e")
    axs[0].set_xticks(range(1, D + 1))
    axs[0].set_xlabel("componente j"); axs[0].set_ylabel("autovalore λ_j (varianza)")
    axs[0].set_title("Autovalori (scree plot)", loc="left")
    axs[1].plot(range(1, D), np.array(errF) * 100, "o-", color="#eb6834", ms=5)
    axs[1].set_xticks(range(1, D))
    axs[1].set_xlabel("rango k"); axs[1].set_ylabel("errore relativo ||X−X_k||_F  [%]")
    axs[1].set_title("Errore dell'approssimazione di rango k", loc="left")
    for a in axs:
        stile(a)
    fig.tight_layout(); fig.savefig("fig_eckart_young.png", dpi=160); plt.close(fig)

    # ---------------------------------------------------------- 2. imputazione dei dati mancanti
    mask = np.isnan(X_nan)
    righe = np.where(mask.any(1))[0]
    print("\n=== 2. Dati mancanti ===")
    for i in righe:
        print(f"  {names[i]}: mancano {', '.join(METRICS[j] for j in np.where(mask[i])[0])}")

    media = np.nanmean(X_nan, 0)
    dev = np.nanstd(X_nan, 0, ddof=1)
    Xs = (X_nan - media) / dev
    Xs[mask] = 0.0                                  # = sostituzione con la media
    vero = ((X_full - media) / dev)[mask]           # valori veri, stessa scala

    rmse_media = np.sqrt(np.mean((0.0 - vero) ** 2))
    print(f"\n  errore (RMSE, in deviazioni standard) sui valori mancanti:")
    print(f"  sostituzione con la media: {rmse_media:.3f}")
    risultati = {}
    for k in range(1, 7):
        Xi, var = imputa_svd(Xs, mask, k)
        rmse = np.sqrt(np.mean((Xi[mask] - vero) ** 2))
        risultati[k] = (rmse, var)
        print(f"  SVD iterativa, k = {k}:    {rmse:.3f}   ({len(var)} iterazioni)")

    # convergenza del punto fisso per k = 3
    var3 = risultati[3][1]
    fatt = np.exp(np.polyfit(np.arange(10, 40), np.log(var3[10:40]), 1)[0]) if len(var3) > 45 else np.nan
    print(f"\n  k = 3: fattore di riduzione della variazione per iterazione = {fatt:.3f}  (convergenza lineare)")

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ks = list(risultati)
    axs[0].axhline(rmse_media, color="#52514e", lw=1.2, ls="--")
    axs[0].text(6.2, rmse_media, "media", va="center", fontsize=8.5, color="#52514e")
    axs[0].plot(ks, [risultati[k][0] for k in ks], "o-", color="#2a78d6", ms=6)
    axs[0].set_xlabel("rango k usato per imputare"); axs[0].set_ylabel("RMSE sui valori mancanti")
    axs[0].set_title("Imputazione: SVD iterativa contro media", loc="left")
    axs[1].semilogy(np.arange(1, len(var3) + 1), var3, color="#2a78d6", lw=2)
    axs[1].set_xlabel("iterazione"); axs[1].set_ylabel("variazione dei valori imputati")
    axs[1].set_title("Punto fisso (k = 3): convergenza lineare", loc="left")
    for a in axs:
        stile(a)
    fig.tight_layout(); fig.savefig("fig_imputazione.png", dpi=160); plt.close(fig)
    print("\ngrafici salvati: fig_eckart_young.png, fig_imputazione.png")

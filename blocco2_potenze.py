"""
BLOCCO 2 - Metodo delle potenze + deflazione
============================================
Teoria:
    v0 = somma_i c_i b_i   ->   C^k v0 = somma_i c_i lambda_i^k b_i
    il termine con lambda_1 domina: errore sull'autovettore ~ |lambda_2/lambda_1|^k
    (C simmetrica: errore sull'autovalore di Rayleigh ~ |lambda_2/lambda_1|^(2k))

Deflazione (vincolo di ortogonalita' della lagrangiana, fatto in pratica):
    C_1 = C - lambda_1 b_1 b_1^T   ha autovalori 0, lambda_2, ..., lambda_D
    -> il metodo delle potenze su C_1 trova b_2, e cosi' via.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from synthetic_data import generate_squad


def metodo_potenze(A, v0, tol=1e-12, max_iter=10_000):
    """Traduzione dello pseudocodice, con un solo prodotto A*v per iterazione."""
    v = v0 / np.linalg.norm(v0)
    lam_old = 0.0
    storia = []                                   # (lambda, v) a ogni iterazione
    for k in range(1, max_iter + 1):
        z = A @ v                                 # 1. nuova direzione (unico prodotto)
        lam_new = v @ z                           # 3. Rayleigh con v vecchio (||v|| = 1)
        v = z / np.linalg.norm(z)                 # 2. normalizzazione
        storia.append((lam_new, v.copy()))
        if abs(lam_new - lam_old) < tol * abs(lam_new):   # 4. arresto relativo
            return lam_new, v, k, storia
        lam_old = lam_new
    print("ATTENZIONE: raggiunto max_iter senza convergenza")
    return lam_new, v, max_iter, storia


def componenti_principali(C, m, seed=0):
    """Prime m componenti con potenze + deflazione."""
    rng = np.random.default_rng(seed)
    A = C.copy()
    lam, B, iters, storie = [], [], [], []
    for j in range(m):
        v0 = rng.standard_normal(C.shape[0])      # iniziale casuale
        l, b, k, st = metodo_potenze(A, v0)
        lam.append(l); B.append(b); iters.append(k); storie.append(st)
        A = A - l * np.outer(b, b)                # deflazione
    return np.array(lam), np.column_stack(B), iters, storie


if __name__ == "__main__":
    _, X_raw, y, roles, names = generate_squad()
    N = X_raw.shape[0]
    X = (X_raw - X_raw.mean(0)) / X_raw.std(0, ddof=1)
    C = X.T @ X / (N - 1)

    m = 3
    lam, B, iters, storie = componenti_principali(C, m)

    # riferimento: autovalori "esatti" di numpy
    lam_ref, V_ref = np.linalg.eigh(C)
    idx = np.argsort(lam_ref)[::-1]
    lam_ref, V_ref = lam_ref[idx], V_ref[:, idx]

    print(" j   lambda potenze    lambda numpy     iterazioni   angolo con b_j di numpy")
    for j in range(m):
        cosang = min(1.0, abs(B[:, j] @ V_ref[:, j]))
        ang = np.degrees(np.arccos(cosang))
        print(f" {j+1}   {lam[j]:12.8f}   {lam_ref[j]:12.8f}    {iters[j]:6d}        {ang:.1e} gradi")

    # ---- verifica della velocita' di convergenza per b1
    rapp = lam_ref[1] / lam_ref[0]
    st = storie[0]
    err_v = np.array([np.sqrt(max(0.0, 1 - (v @ V_ref[:, 0])**2)) for _, v in st])  # seno dell'angolo
    err_l = np.array([abs(l - lam_ref[0]) for l, _ in st])
    k = np.arange(1, len(st) + 1)

    # pendenza misurata sul tratto centrale (log dell'errore contro k)
    sel = (err_v > 1e-13) & (k > 5)
    fattore = np.exp(np.polyfit(k[sel], np.log(err_v[sel]), 1)[0])
    print(f"\n|lambda2/lambda1| teorico = {rapp:.4f}")
    print(f"fattore di riduzione misurato per iterazione (autovettore) = {fattore:.4f}")

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.semilogy(k, err_v, color="#2a78d6", lw=2, label="errore autovettore (seno dell'angolo)")
    ax.semilogy(k, err_l, color="#eb6834", lw=2, label="errore autovalore (Rayleigh)")
    kk = np.arange(1, len(st) + 1)
    ax.semilogy(kk, err_v[19] * rapp ** (kk - 20), color="#2a78d6", lw=1, ls="--",
                label=r"teoria: $|\lambda_2/\lambda_1|^k$")
    ax.semilogy(kk, err_l[19] * rapp ** (2 * (kk - 20)), color="#eb6834", lw=1, ls="--",
                label=r"teoria: $|\lambda_2/\lambda_1|^{2k}$")
    ax.set_ylim(1e-16, 10)
    ax.set_xlabel("iterazione k"); ax.set_ylabel("errore")
    ax.set_title("Metodo delle potenze su C: convergenza a b1", loc="left")
    ax.grid(True, color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig("fig_potenze_convergenza.png", dpi=160)
    print("grafico salvato: fig_potenze_convergenza.png")

"""
BLOCCO 3 - Gruppi di lavoro bilanciati e direzioni di lavoro
=============================================================
Teoria:
    score del giocatore n nello spazio latente:  z_n = B^T x_n   (k componenti)
    media della squadra = 0 (dati centrati)

    SS_tot    = somma_n ||z_n||^2                         (fissa)
    SS_tra    = somma_g n_g ||zbar_g||^2                  (gruppi diversi tra loro)
    SS_dentro = somma_g somma_{n in g} ||z_n - zbar_g||^2 (varieta' dentro il gruppo)
    SS_tot = SS_tra + SS_dentro

    Obiettivo: min SS_tra  <=>  max SS_dentro  (forti e deboli nello stesso gruppo).
    Gli assi non vanno ripesati: lo score lungo b_j ha gia' varianza lambda_j.

Algoritmo (euristica di scambio):
    parto da gruppi casuali; provo tutti gli scambi di due giocatori di gruppi
    diversi; applico lo scambio che fa scendere di piu' SS_tra; ripeto finche'
    nessuno scambio migliora (minimo locale). Ripeto da piu' partenze casuali.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from synthetic_data import generate_squad, METRICS
from blocco2_potenze import componenti_principali

K = 3            # componenti tenute (circa 80% della varianza)
G = 5            # numero di gruppi, da 5 giocatori
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


# ------------------------------------------------------------------ decomposizione
def decomposizione(Z, gruppi):
    """Ritorna SS_tot, SS_tra, SS_dentro (Z centrato: media della squadra = 0)."""
    ss_tot = np.sum(Z**2)
    ss_tra, ss_dentro = 0.0, 0.0
    for g in np.unique(gruppi):
        Zg = Z[gruppi == g]
        media_g = Zg.mean(axis=0)
        ss_tra += len(Zg) * media_g @ media_g
        ss_dentro += np.sum((Zg - media_g) ** 2)
    return ss_tot, ss_tra, ss_dentro


def ss_tra(Z, gruppi):
    return decomposizione(Z, gruppi)[1]


# ------------------------------------------------------------------ ottimizzazione
def bilancia(Z, G, n_partenze=30, seed=0):
    rng = np.random.default_rng(seed)
    N = len(Z)
    migliore, val_migliore = None, np.inf
    for _ in range(n_partenze):
        gruppi = rng.permutation(np.arange(N) % G)          # gruppi casuali, 5 per gruppo
        val = ss_tra(Z, gruppi)
        while True:
            best_delta, best_scambio = 0.0, None
            for i in range(N):
                for j in range(i + 1, N):
                    if gruppi[i] == gruppi[j]:
                        continue
                    gruppi[i], gruppi[j] = gruppi[j], gruppi[i]      # provo lo scambio
                    delta = ss_tra(Z, gruppi) - val
                    gruppi[i], gruppi[j] = gruppi[j], gruppi[i]      # lo annullo
                    if delta < best_delta - 1e-12:
                        best_delta, best_scambio = delta, (i, j)
            if best_scambio is None:                                  # minimo locale
                break
            i, j = best_scambio
            gruppi[i], gruppi[j] = gruppi[j], gruppi[i]
            val += best_delta
        if val < val_migliore:
            migliore, val_migliore = gruppi.copy(), val
    return migliore, val_migliore


# ------------------------------------------------------------------ direzioni di lavoro
def direzioni_di_lavoro(X, Z, lam, B, roles, names, gruppi, assi=(0, 1), soglia=0.25):
    """
    Gli assi descrivono STILI, non livelli: confronto ogni giocatore con la media del
    suo ruolo.
    1) Asse: su b_j il verso "tipico del ruolo" e' s = segno(media del ruolo). Uso l'asse
       solo se il ruolo sta chiaramente da una parte (|media| >= 0.5 sqrt(lambda_j)).
           deficit_j = s * (z_j - media_ruolo_j) / sqrt(lambda_j)
       Scelgo l'asse con il deficit piu' negativo; sotto -soglia e' un margine di lavoro.
    2) Statistiche: tra quelle con peso grande nel verso s di b_j, tengo solo quelle
       che il ruolo usa davvero (media del ruolo sopra la media della squadra) e in cui
       il giocatore e' sotto la media del suo ruolo.
    3) Mentore: il compagno del suo gruppo piu' forte su quelle statistiche.
    """
    N = len(Z)
    idx_ruolo = {r: [i for i in range(N) if roles[i] == r] for r in set(roles)}
    mz = {r: Z[ix].mean(axis=0) for r, ix in idx_ruolo.items()}      # media ruolo, spazio latente
    mx = {r: X[ix].mean(axis=0) for r, ix in idx_ruolo.items()}      # media ruolo, statistiche
    righe = []
    for n in range(N):
        r = roles[n]
        migliore = None
        for j in assi:
            if abs(mz[r][j]) < 0.5 * np.sqrt(lam[j]):
                continue                        # il ruolo non e' caratterizzato da questo asse
            s = np.sign(mz[r][j])
            deficit = s * (Z[n, j] - mz[r][j]) / np.sqrt(lam[j])
            if migliore is None or deficit < migliore[1]:
                migliore = (j, deficit, s)
        if migliore is None or migliore[1] > -soglia:
            righe.append((names[n], "-", "in linea o sopra la media del ruolo", "-"))
            continue
        j, deficit, s = migliore
        ordine = np.argsort(s * B[:, j])[::-1]
        stat = [t for t in ordine[:6] if mx[r][t] > 0 and X[n, t] < mx[r][t]][:3]
        if not stat:
            stat = [ordine[0]]
        compagni = [i for i in range(N) if gruppi[i] == gruppi[n] and i != n]
        mentore = max(compagni, key=lambda i: X[i, stat].mean())
        righe.append((names[n], f"b{j+1} ({deficit:+.2f})",
                      ", ".join(METRICS[t] for t in stat), names[mentore]))
    return righe


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    _, X_raw, y, roles, names = generate_squad()
    N = X_raw.shape[0]
    X = (X_raw - X_raw.mean(0)) / X_raw.std(0, ddof=1)
    C = X.T @ X / (N - 1)
    lam, B, _, _ = componenti_principali(C, K)

    # convenzione sul segno (b e -b sono equivalenti): il peso piu' grande in modulo e' positivo
    for j in range(K):
        if B[np.argmax(np.abs(B[:, j])), j] < 0:
            B[:, j] *= -1
    Z = X @ B
    print("assi: b1 = contatto (+) / gioco al largo (-)")
    print("      b2 = portatore di palla (+) / chi porta poco palla: registi e piloni (-)")
    print(f"varianza spiegata con K={K}: {lam.sum() / np.trace(C):.1%}\n")

    # --- ottimizzazione
    gruppi, val = bilancia(Z, G)
    ss_tot, ss_t, ss_d = decomposizione(Z, gruppi)
    print("=== Decomposizione della varianza (gruppi ottimizzati) ===")
    print(f"  SS_tot = {ss_tot:.4f}   SS_tra = {ss_t:.4f}   SS_dentro = {ss_d:.4f}")
    print(f"  controllo SS_tra + SS_dentro - SS_tot = {ss_t + ss_d - ss_tot:.1e}")
    print(f"  quota tra i gruppi: {ss_t / ss_tot:.2%}")

    # --- confronto con gruppi casuali
    rng = np.random.default_rng(1)
    casuali = np.array([ss_tra(Z, rng.permutation(np.arange(N) % G)) for _ in range(5000)])
    print("\n=== Confronto con 5000 divisioni casuali ===")
    print(f"  SS_tra casuale: media {casuali.mean():.3f}, minimo {casuali.min():.3f}")
    print(f"  SS_tra ottimizzato: {val:.4f}  ({val / casuali.mean():.1%} della media casuale)")
    print(f"  divisioni casuali migliori dell'ottimizzata: {(casuali <= val).mean():.2%}")

    # --- composizione dei gruppi
    print("\n=== Gruppi ===")
    for g in range(G):
        idx = np.where(gruppi == g)[0]
        media = Z[idx].mean(axis=0)
        print(f"  Gruppo {g+1}  media (b1,b2,b3) = ({media[0]:+.2f}, {media[1]:+.2f}, {media[2]:+.2f})")
        print("     " + ", ".join(names[i] for i in idx))

    # --- direzioni di lavoro
    print("\n=== Direzioni di lavoro (confronto con il proprio ruolo) ===")
    print(f"  {'giocatore':18s} {'asse':12s} {'da allenare':52s} mentore")
    for nome, asse, top, mentore in direzioni_di_lavoro(X, Z, lam, B, roles, names, gruppi):
        print(f"  {nome:18s} {asse:12s} {top:52s} {mentore}")

    # --- grafico 1: giocatori sul piano b1-b2, colorati per gruppo
    abbr = {"Pilone": "PIL", "Tallonatore": "TAL", "Seconda linea": "2L", "Terza linea": "3L",
            "Mediano mischia": "MM", "Apertura": "AP", "Centro": "CE", "Ala": "ALA", "Estremo": "EST"}
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for g in range(G):
        idx = np.where(gruppi == g)[0]
        ax.scatter(Z[idx, 0], Z[idx, 1], s=70, color=COL[g], edgecolor="white", lw=1.5,
                   label=f"Gruppo {g+1}", zorder=3)
        m = Z[idx, :2].mean(axis=0)
        ax.scatter(*m, marker="X", s=110, color=COL[g], edgecolor="white", lw=1.2, zorder=4)
    for n in range(N):
        ax.annotate(abbr[roles[n]], (Z[n, 0], Z[n, 1]), xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, color="#52514e")
    ax.axhline(0, color="#c3c2b7", lw=0.8); ax.axvline(0, color="#c3c2b7", lw=0.8)
    ax.set_xlabel("b1: gioco al largo  ←→  contatto")
    ax.set_ylabel("b2: poco portatore (registi, piloni)  ←→  portatore")
    ax.set_title("Giocatori nello spazio degli stili; X = media del gruppo", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=5, bbox_to_anchor=(0, -0.12))
    fig.tight_layout(); fig.savefig("fig_gruppi.png", dpi=160); plt.close(fig)

    # --- grafico 2: ottimizzato contro casuali
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.hist(casuali, bins=50, color="#86b6ef", edgecolor="white", lw=0.6)
    ax.axvline(val, color="#eb6834", lw=2)
    ax.annotate(f"gruppi ottimizzati\nSS_tra = {val:.2f}", xy=(val, ax.get_ylim()[1] * 0.8),
                xytext=(12, 0), textcoords="offset points", fontsize=9, color="#0b0b0b")
    ax.set_xlabel("SS_tra (squilibrio tra i gruppi)"); ax.set_ylabel("numero di divisioni casuali")
    ax.set_title("5000 divisioni casuali contro quella ottimizzata", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig("fig_confronto_casuali.png", dpi=160); plt.close(fig)
    print("\ngrafici salvati: fig_gruppi.png, fig_confronto_casuali.png")

"""
Generatore di un dataset sintetico realistico per una rosa di rugby a XV.

Ogni riga e' un giocatore, ogni colonna una metrica stagionale normalizzata
su 80 minuti. I profili medi per ruolo sono costruiti a mano in modo che:
  - Piloni, Seconde linee, Ali, Mediani siano "polarizzati" (lontani dal confine);
  - Terze linee e Centri siano "ibridi" (vicini al confine Avanti/Trequarti).
Questo permette di verificare se la SVM individua come Support Vectors
proprio i ruoli ibridi attesi.
"""
import numpy as np

METRICS = [
    "metri_guadagnati",
    "placcaggi_dominanti",
    "ruck_offensive",
    "passaggi",
    "offload",
    "difensori_battuti",
    "salti_touche",
    "calci_in_gioco",
    "placcaggi_effettuati",
    "portate",
]

# Profili medi per 80 minuti (stesso ordine di METRICS)
ROLE_PROFILES = {
    #                  metri plD  ruck  pass  off  difB  touc  calci plac  port
    "Pilone":         [18,   1.0, 15.0,  3,   0.3, 0.6,  0.2,  0.0, 10,   7],
    "Tallonatore":    [24,   1.2, 13.0,  5,   0.5, 1.0,  0.1,  0.0, 12,   8],
    "Seconda linea":  [22,   1.4, 14.0,  4,   0.4, 0.8,  6.5,  0.0, 13,   8],
    "Terza linea":    [44,   1.8,  9.0,  8,   1.1, 2.4,  1.5,  0.1, 12,  11],
    "Mediano mischia": [28,  0.4,  2.5, 65,   0.3, 1.5,  0.0,  4.0,  6,   4],
    "Apertura":       [34,   0.5,  2.5, 28,   0.5, 2.0,  0.0, 12.0,  7,   6],
    "Centro":         [50,   1.4,  6.0, 10,   1.2, 3.2,  0.0,  0.8, 11,  11],
    "Ala":            [72,   0.3,  2.5,  5,   0.8, 4.6,  0.0,  1.5,  5,   9],
    "Estremo":        [66,   0.3,  2.5,  7,   0.7, 4.0,  0.0,  6.0,  5,   9],
}

# Composizione della rosa: 14 Avanti + 11 Trequarti = 25
SQUAD = [
    ("Pilone", 4), ("Tallonatore", 2), ("Seconda linea", 3), ("Terza linea", 5),
    ("Mediano mischia", 2), ("Apertura", 2), ("Centro", 3), ("Ala", 3), ("Estremo", 1),
]

FORWARDS = {"Pilone", "Tallonatore", "Seconda linea", "Terza linea"}
HYBRID_ROLES = {"Terza linea", "Centro"}


def generate_squad(seed=7, rel_noise=0.18, missing_players=2, missing_frac=0.4):
    """
    Genera la rosa sintetica.

    Parametri
    ---------
    seed : int
        Seme del generatore casuale (riproducibilita').
    rel_noise : float
        Deviazione standard relativa del rumore rispetto al profilo medio.
    missing_players : int
        Numero di giocatori che hanno "saltato meta' stagione": su di loro
        una frazione delle metriche viene posta a NaN.
    missing_frac : float
        Frazione di metriche mancanti per ciascuno di quei giocatori.

    Ritorna
    -------
    X : ndarray (25, p)      dati con NaN
    X_full : ndarray (25, p) dati completi (ground truth per l'imputazione)
    y : ndarray (25,)        +1 Avanti, -1 Trequarti
    roles : list[str]        ruolo ufficiale di ciascun giocatore
    names : list[str]        identificativo del giocatore
    """
    rng = np.random.default_rng(seed)
    roles, rows = [], []
    for role, count in SQUAD:
        mu = np.asarray(ROLE_PROFILES[role], dtype=float)
        for _ in range(count):
            # rumore moltiplicativo + piccola componente additiva, troncato a >= 0
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

    # Giocatori con meta' stagione saltata: NaN su alcune metriche
    X = X_full.copy()
    p = X.shape[1]
    n_miss = max(1, int(round(missing_frac * p)))
    for i in rng.choice(len(roles), size=missing_players, replace=False):
        cols = rng.choice(p, size=n_miss, replace=False)
        X[i, cols] = np.nan

    return X, X_full, y, roles, names

"""
BLOCCO 1 - Dati, matrice di covarianza, componenti principali
=============================================================
Teoria (dalla lagrangiana):
    max b^T C b  con  b^T b = 1   ->   C b = lambda b
    varianza lungo b_i = lambda_i
    varianza totale    = somma degli autovalori = traccia(C)

Qui calcoliamo le componenti in due modi e controlliamo che coincidano:
    (a) autovalori di C = X^T X / (N-1)
    (b) SVD di X:  X = U S V^T   ->   lambda_i = s_i^2 / (N-1),  b_i = colonne di V
"""
import numpy as np
from synthetic_data import generate_squad, METRICS

# ---------------------------------------------------------------- 1. dati
# X_full: dati completi (i NaN li gestiamo piu' avanti)
_, X_raw, y, roles, names = generate_squad()
N, D = X_raw.shape                       # N = 25 giocatori, D = 10 statistiche

# ---------------------------------------------------------------- 2. standardizzazione
media = X_raw.mean(axis=0)
dev_std = X_raw.std(axis=0, ddof=1)
X = (X_raw - media) / dev_std            # ogni colonna: media 0, varianza 1

# ---------------------------------------------------------------- 3. matrice di covarianza
C = X.T @ X / (N - 1)                    # D x D; con dati standardizzati = correlazione
print("C e' simmetrica?        ", np.allclose(C, C.T))
print("diagonale di C (=1?):   ", np.round(np.diag(C), 6))
print("traccia(C) =", round(np.trace(C), 6), "  (= D =", D, ")")

# ---------------------------------------------------------------- 4a. autovalori di C
lam, B = np.linalg.eigh(C)               # eigh: per matrici simmetriche, ordine crescente
ordine = np.argsort(lam)[::-1]           # riordino dal piu' grande
lam, B = lam[ordine], B[:, ordine]

# ---------------------------------------------------------------- 4b. SVD di X
U, s, Vt = np.linalg.svd(X, full_matrices=False)
lam_svd = s**2 / (N - 1)

print("\n  i   lambda (eig C)   s^2/(N-1) (SVD)   varianza spiegata   cumulata")
cum = np.cumsum(lam) / lam.sum()
for i in range(D):
    print(f" {i+1:2d}   {lam[i]:12.6f}   {lam_svd[i]:14.6f}     {lam[i]/lam.sum():8.1%}        {cum[i]:6.1%}")

print("\nsomma autovalori =", round(lam.sum(), 6), " = traccia(C)")
# autovettori: uguali a meno del segno (b e -b sono entrambi autovettori)
segni = np.sign(np.sum(B * Vt.T, axis=0))
print("autovettori eig vs SVD, max differenza (a meno del segno):",
      f"{np.max(np.abs(B - Vt.T * segni)):.2e}")

# ---------------------------------------------------------------- 5. le prime due componenti
print("\nPesi (loadings) delle prime due componenti:")
print(f"  {'statistica':22s} {'b1':>7s} {'b2':>7s}")
for j, nome in enumerate(METRICS):
    print(f"  {nome:22s} {B[j,0]:7.2f} {B[j,1]:7.2f}")

# ---------------------------------------------------------------- 6. score dei giocatori
Z = X @ B[:, :2]                          # z_n = B^T x_n per ogni giocatore
print("\nvarianza degli score lungo b1 e b2:", np.round(Z.var(axis=0, ddof=1), 6),
      " (= lambda1, lambda2)")

# ---------------------------------------------------------------- 7. perche' si centra prima (MML, par. 6.4.3)
# V[x] = E[(x - mu)^2]  (due passate: prima la media, poi gli scarti)
#      = E[x^2] - (E[x])^2  (una passata: stessa cosa in aritmetica esatta)
# In floating point la seconda sottrae due numeri enormi e quasi uguali:
# cancellazione catastrofica. Esempio: 4 misure con un offset grande.
print("\nVarianza di [4, 7, 13, 16] + offset (valore esatto 22.5):")
for offset in [0.0, 1e8, 1e9]:
    x = offset + np.array([4.0, 7.0, 13.0, 16.0])
    una_passata = np.mean(x**2) - np.mean(x)**2
    due_passate = np.mean((x - x.mean())**2)
    print(f"  offset {offset:8.0e}:  E[x^2]-E[x]^2 = {una_passata:8.1f}    E[(x-mu)^2] = {due_passate:6.1f}")

"""
BLOCK 1 - Data, covariance matrix, principal components
=======================================================
Theory (from the Lagrangian):
    max b^T C b  subject to  b^T b = 1   ->   C b = lambda b
    variance along b_i = lambda_i
    total variance     = sum of the eigenvalues = trace(C)

Here we compute the components in two ways and check that they agree:
    (a) eigenvalues of C = X^T X / (N-1)
    (b) SVD of X:  X = U S V^T   ->   lambda_i = s_i^2 / (N-1),  b_i = columns of V
"""
import numpy as np
from synthetic_data import generate_squad, METRICS

# ---------------------------------------------------------------- 1. data
# X_full: complete data (the NaNs are handled later)
_, X_raw, y, roles, names = generate_squad()
N, D = X_raw.shape                       # N = 25 players, D = 10 statistics

# ---------------------------------------------------------------- 2. standardisation
mean = X_raw.mean(axis=0)
std_dev = X_raw.std(axis=0, ddof=1)
X = (X_raw - mean) / std_dev             # each column: mean 0, variance 1

# ---------------------------------------------------------------- 3. covariance matrix
C = X.T @ X / (N - 1)                    # D x D; with standardised data = correlation
print("is C symmetric?         ", np.allclose(C, C.T))
print("diagonal of C (=1?):    ", np.round(np.diag(C), 6))
print("trace(C) =", round(np.trace(C), 6), "  (= D =", D, ")")

# ---------------------------------------------------------------- 4a. eigenvalues of C
lam, B = np.linalg.eigh(C)               # eigh: for symmetric matrices, ascending order
order = np.argsort(lam)[::-1]            # reorder from the largest
lam, B = lam[order], B[:, order]

# ---------------------------------------------------------------- 4b. SVD of X
U, s, Vt = np.linalg.svd(X, full_matrices=False)
lam_svd = s**2 / (N - 1)

print("\n  i   lambda (eig C)   s^2/(N-1) (SVD)   explained variance   cumulative")
cum = np.cumsum(lam) / lam.sum()
for i in range(D):
    print(f" {i+1:2d}   {lam[i]:12.6f}   {lam_svd[i]:14.6f}     {lam[i]/lam.sum():8.1%}         {cum[i]:6.1%}")

print("\nsum of eigenvalues =", round(lam.sum(), 6), " = trace(C)")
# eigenvectors: equal up to the sign (b and -b are both eigenvectors)
signs = np.sign(np.sum(B * Vt.T, axis=0))
print("eigenvectors eig vs SVD, max difference (up to the sign):",
      f"{np.max(np.abs(B - Vt.T * signs)):.2e}")

# ---------------------------------------------------------------- 5. the first two components
print("\nWeights (loadings) of the first two components:")
print(f"  {'statistic':22s} {'b1':>7s} {'b2':>7s}")
for j, name in enumerate(METRICS):
    print(f"  {name:22s} {B[j,0]:7.2f} {B[j,1]:7.2f}")

# ---------------------------------------------------------------- 6. player scores
Z = X @ B[:, :2]                          # z_n = B^T x_n for each player
print("\nvariance of the scores along b1 and b2:", np.round(Z.var(axis=0, ddof=1), 6),
      " (= lambda1, lambda2)")

# ---------------------------------------------------------------- 7. why we centre first (MML, sec. 6.4.3)
# V[x] = E[(x - mu)^2]  (two passes: first the mean, then the deviations)
#      = E[x^2] - (E[x])^2  (one pass: the same thing in exact arithmetic)
# In floating point the second subtracts two huge, nearly equal numbers:
# catastrophic cancellation. Example: 4 measurements with a large offset.
print("\nVariance of [4, 7, 13, 16] + offset (exact value 22.5):")
for offset in [0.0, 1e8, 1e9]:
    x = offset + np.array([4.0, 7.0, 13.0, 16.0])
    one_pass = np.mean(x**2) - np.mean(x)**2
    two_pass = np.mean((x - x.mean())**2)
    print(f"  offset {offset:8.0e}:  E[x^2]-E[x]^2 = {one_pass:8.1f}    E[(x-mu)^2] = {two_pass:6.1f}")

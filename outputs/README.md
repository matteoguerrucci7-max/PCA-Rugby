# Outputs

Console output of each script, saved so the numbers reported in the main README can be checked without running the code.

| File | Script | Content |
| --- | --- | --- |
| `block1_output.txt` | `block1_pca_svd.py` | Covariance, SVD, explained variance, axis weights |
| `block2_output.txt` | `block2_power_method.py` | Power method with deflation, convergence |
| `block3_output.txt` | `block3_groups.py` | Balanced groups, comparison with random splits, work directions |
| `block4_output.txt` | `block4_rank_imputation.py` | Eckart–Young, missing-data imputation |
| `block5_output.txt` | `block5_shift_sensitivity.py` | Shift, inverse power method (LU), noise sensitivity |

To regenerate, e.g.: `python block3_groups.py > outputs/block3_output.txt`

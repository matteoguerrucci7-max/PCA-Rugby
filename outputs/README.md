# Outputs

Console output of each script, saved so the numbers reported in the main README can be checked without running the code.

| File | Script | Content |
| --- | --- | --- |
| `output_block1.txt` | `block1_pca_svd.py` | Covariance, SVD, explained variance, axis weights |
| `output_block2.txt` | `block2_power_method.py` | Power method with deflation, convergence |
| `output_block3.txt` | `block3_groups.py` | Balanced groups, comparison with random splits, work directions |
| `output_block4.txt` | `block4_rank_imputation.py` | Eckart–Young, missing-data imputation |
| `output_block5.txt` | `block5_shift_sensitivity.py` | Shift, inverse power method (LU), noise sensitivity |

To regenerate, e.g.: `python block3_groups.py > outputs/output_block3.txt`

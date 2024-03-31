# Optimal Ridge Regularization for Out-of-Distribution Prediction

## Scripts

The following files are included in this repository:
- Util functions:
    - `generate_data.py`: A Python script that generates the data set.
    - `compute_risk.py`: A Python script that computes the empirical and theoretical risks of ridge predictor and ensembles.
    - `fixed_point_sol.py`: A Python script that computes the fixed-point solutions in $(\lambda,\phi)$.
- Ex1
    - `ex1_equiv_lam_min.py`: compute the minimum feasible $\lambda$.
    - `ex1_opt_ridge.py`: compute the in-distribution risk of ridge predictors.
    - `ex1_opt_ridge_ood.py`: compute the OOD risk of ridge predictors.
- Ex2
    - `ex2_MNIST.py`: compute the OOD risk with distribution shifts on MNIST datasets.
- Ex3
    - `ex3_mono.py`: compute the ridge risk at different values of $\lambda$.
    - `ex3_MNIST.py`: compute the OOD risk at different values of $\lambda$ on MNIST datasets.
- Ex4
    - `ex4_equiv_v.py`: compute the fixed-point solutions in $(\lambda,\phi)$.
    - `ex4_equiv_risk.py`: compute the risks of ridge predictors in $(\lambda,\phi)$.
- Ex5: Figure F8
    - `ex5_theory_ridge_opt.py`: theoretical risk of ridge predictors.
    - `ex5_theory_ridgeless.py`: theoretical risk of full-ensemble ridgeless predictors.
- Visualization:
    - `Plot.ipynb`: A Jupyter notebook that visualizes the results.


## Computation details

All the experiments are run on Ubuntu 22.04.2 LTS (GNU/Linux 5.15.0-72-generic x86_64) using 12 cores.

The estimated time to run all experiments is roughly less than 2 hours for each script.    
import os
import pandas as pd
import numpy as np
from fixed_point_sol import *
from generate_data import ar1_cov
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from joblib import Parallel, delayed


def comp_risk_inf(
    Sigma, beta0, sigma2, lam, phi, psi, ATA=None):
    v = v_general(psi, lam, Sigma)
    tv = tv_general(phi, psi, lam, Sigma, v, ATA)
    V = (1 + tv) * sigma2
    B = tc_general(phi, psi, lam, Sigma, beta0, v, tv, ATA)
    R = B+V

    return R



p = 1000

cov = 'ar1'
rho_ar1 = 0.25
Sigma = ar1_cov(rho_ar1, p)

ATA = Sigma

top_k = 1
_, beta0 = eigsh(Sigma, k=top_k)
rho2 = (1-rho_ar1**2)/(1-rho_ar1)**2/top_k
beta0 = np.mean(beta0, axis=-1) * 10      

sigma = 0.5
sigma2 = sigma**2


path_result = 'result/ex5/ridge/{:.02f}_{:.02f}/'.format(rho_ar1, sigma)
os.makedirs(path_result, exist_ok=True)


def run_one_simulation(phi, psi, lam):
    if psi<phi:
        R = np.nan
    else:
        R = comp_risk_inf(Sigma, beta0, sigma2, lam, phi, psi, ATA)
    return [phi,psi,lam,R]


phi_list = np.logspace(-1,1,100)
lam_list = np.r_[0., np.logspace(-1,2,100)]
with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:        
    res = parallel(
        delayed(run_one_simulation)(phi, phi, lam) 
        for phi in tqdm(phi_list, desc='phi') for lam in tqdm(lam_list, desc='lam')
    )
df = pd.DataFrame(np.array(res), columns=['phi','psi','lam','risk'])
df.to_pickle(path_result+'res_ridge_opt_risk.pkl', compression='gzip')
_df = df.pivot(index="phi", columns="lam", values='risk')
_df.to_pickle(path_result+'res_opt_ridge.pkl', compression='gzip')
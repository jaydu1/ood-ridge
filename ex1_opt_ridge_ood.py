import os
import pandas as pd
import numpy as np
from fixed_point_sol import *
from generate_data import ar1_cov
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from joblib import Parallel, delayed


def comp_risk_inf(
    Sigma, beta, sigma2, lam, phi, psi, Sigma0=None, beta0=None):
    v = v_general(psi, lam, Sigma)
    tv = tv_general(phi, psi, lam, Sigma, v, Sigma0)
    V = (1 + tv) * sigma2
    B = tc_general(phi, psi, lam, Sigma, beta, v, tv, Sigma0)
    R = B+V

    if Sigma0 is None:
        Sigma0 = Sigma
    if beta0 is not None:
        E = 2 * beta.T @ np.linalg.solve(np.identity(p) + v*Sigma, Sigma0 @ (beta0-beta))
        kappa = (beta0-beta).T @ Sigma0 @ (beta0-beta)
        R_out = R + E + kappa
    else:
        R_out = R

    tv = tv_general(phi, psi, lam, Sigma, v, None)
    V = (1 + tv) * sigma2
    B = tc_general(phi, psi, lam, Sigma, beta, v, tv, None)
    R_in = B+V

    return R_in, R_out


p = 500

cov = 'ar1'

shift = 'sig'#'cov'#
if shift=='cov':
    rho_ar1 = 0.
    rho_ar1_0 = 0.5
    Sigma = ar1_cov(rho_ar1, p)
    Sigma0 = ar1_cov(rho_ar1_0, p)

    top_k = 1
    _, beta0_1 = eigsh(Sigma0, k=top_k, which='LM')
    _, beta0_2 = eigsh(Sigma0, k=top_k, which='SM')
    beta = np.mean(beta0_1 + beta0_2, axis=-1)/2
    beta0 = None

    phi_list = [1.5, 2, 3, 4]
else:
    rho_ar1 = 0.5
    Sigma = ar1_cov(rho_ar1, p)
    Sigma0 = None
    
    top_k = 1
    _, beta0_1 = eigsh(Sigma, k=top_k, which='LM')
    _, beta0_2 = eigsh(Sigma, k=top_k, which='SM')
    beta = np.mean(beta0_1 + beta0_2, axis=-1)/2
    beta0 = 2*beta
    # beta0 = -beta

    phi_list = [0.5]

sigma = .1
sigma2 = sigma**2


 

psi_list = np.logspace(-1,1,1000)
psi_list = np.unique(np.round(psi_list, 6))

arr_lam_min = - (-1 + np.sqrt(psi_list))**2
lam_list = np.r_[
    -np.logspace(-3, np.log10(-np.min(arr_lam_min)), 400)[-1::-1][::2],
    [0.], np.logspace(-3,1,1000)[::2]
]
lam_list = np.unique(np.round(lam_list, 6))

path_result = 'result/ex1/'
filename = path_result+'res_equiv_lam_min_{:.02f}.pkl'.format(rho_ar1)
_df = pd.read_pickle(filename, compression='gzip')
dict_lam_min = {i[0]:i[1] for i in _df.values}





path_result = 'result/ex1/{:.02f}_{:.02f}/'.format(rho_ar1, sigma)
os.makedirs(path_result, exist_ok=True)


def run_one_simulation(phi, psi, lam, lam_min):
    if psi<phi or lam < lam_min:
        R_in = R_out = np.nan
    else:
        R_in, R_out = comp_risk_inf(Sigma, beta, sigma2, lam, phi, psi, Sigma0, beta0)
        R_in = np.round(R_in, 8)
        R_out = np.round(R_out, 8)
    return [phi,psi,lam,R_in, R_out]

    

filename = path_result+'res_opt_ridge_{}.pkl'.format(shift)
with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:
    df = pd.DataFrame(columns=['phi','lam','risk_in','risk_out'])
    for phi in tqdm(phi_list, desc='phi'):
        psi = psi_list[np.argmin(np.abs(psi_list - phi))]
        res = parallel(
            delayed(run_one_simulation)(psi, psi, lam, dict_lam_min[psi]) 
            for lam in tqdm(lam_list, desc='lam')
        )
        _df = pd.DataFrame(np.array(res), columns=['phi','psi','lam','risk_in','risk_out'])
        df = pd.concat([df, _df], axis=0)
        df.to_pickle(filename, compression='gzip')
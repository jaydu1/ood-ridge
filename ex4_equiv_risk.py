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


p = 500

cov = 'ar1'
rho_ar1 = 0.25
Sigma = ar1_cov(rho_ar1, p)

top_k = 1
_, beta0 = eigsh(Sigma, k=top_k)
rho2 = (1-rho_ar1**2)/(1-rho_ar1)**2/top_k
beta0 = np.mean(beta0, axis=-1)

sigma = .1
sigma2 = sigma**2


phi = 0.5; psi_list = np.logspace(-1,1,1000)[::2]
psi_list = psi_list[psi_list>=phi]
psi_list = np.unique(np.round(psi_list, 6))

arr_lam_min = - (-1 + np.sqrt(psi_list))**2
lam_list = np.r_[
    -np.logspace(-3, np.log10(-np.min(arr_lam_min)), 400)[-1::-1][::2],
    [0.], np.logspace(-3,1,1000)[::2]
]


path_result = 'result/ex1/'
filename = path_result+'res_equiv_lam_min_{:.02f}.pkl'.format(rho_ar1)
_df = pd.read_pickle(filename, compression='gzip')
dict_lam_min = {i[0]:i[1] for i in _df.values}

lam_list = np.unique(np.round(lam_list, 6))



path_result = 'result/ex4/{:.02f}_{:.02f}_{:.02f}/'.format(phi, rho_ar1, sigma)
os.makedirs(path_result, exist_ok=True)


def run_one_simulation(phi, psi, lam, lam_min):
    if psi<phi or lam < lam_min:
        R = np.nan
    else:
        R = comp_risk_inf(Sigma, beta0, sigma2, lam, phi, psi)
        R = np.round(R, 8)
    return [phi,psi,lam,R]


def load_or_initialize(filename, columns):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return pd.DataFrame(columns=columns)

def append_to_file(df, filename):
    df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

    

filename = path_result+'res_equiv_risk.pkl'
with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:
    df = pd.DataFrame(columns=['phi','psi','lam','risk'])
    for lam in tqdm(lam_list, desc='lam'):
        res = parallel(
            delayed(run_one_simulation)(phi, psi, lam, dict_lam_min[psi]) 
            for psi in tqdm(psi_list, desc='psi')
        )
        _df = pd.DataFrame(np.array(res), columns=['phi','psi','lam','risk'])
        df = pd.concat([df, _df], axis=0)
        df.to_pickle(filename, compression='gzip')
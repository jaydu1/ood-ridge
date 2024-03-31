import os
import pandas as pd
import numpy as np
from fixed_point_sol import *
from generate_data import ar1_cov
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from joblib import Parallel, delayed



p = 500

cov = 'ar1';rho_ar1 = 0.5;Sigma = ar1_cov(rho_ar1, p);r_min = np.linalg.eigh(Sigma)[0][0]
cov = 'ar1';rho_ar1 = 0.;Sigma = None;r_min=1.


phi = 0.1; psi_list = np.logspace(-1,1,1000)
psi_list = psi_list[psi_list>=phi]
psi_list = np.unique(np.round(psi_list, 6))

path_result = 'result/ex1/'
os.makedirs(path_result, exist_ok=True)


def run_one_simulation(psi, Sigma):
    lam_min = lam_min_general(psi, Sigma, r_min)
    return [psi,lam_min]


filename = path_result+'res_equiv_lam_min_{:.02f}.pkl'.format(rho_ar1)
with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:
    res = parallel(
        delayed(run_one_simulation)(psi, Sigma) for psi in tqdm(psi_list, desc='psi')
    )
    df = pd.DataFrame(np.array(res), columns=['phi','lam_min'])
    df.to_pickle(filename, compression='gzip')
    
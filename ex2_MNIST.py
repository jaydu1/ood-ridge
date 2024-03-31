import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from joblib import Parallel, delayed

mnist = fetch_openml('mnist_784')
X = mnist.data.values
y = mnist.target.astype('int').values
print(X.shape)

X = X/255*2 - 1

y_train = y[:60000]
y_test = y[60000:]
X_train = X[:60000,:]
X_test = X[60000:,:]


y_test_ood = y[60000:].copy()
X_test_ood = X[60000:,:]



p = 500
np.random.seed(42)
W = np.random.randn(784,p) / 5

X_rand_train = np.exp(-1j * X_train@W)
X_rand_train = np.concatenate((np.real(X_rand_train), np.imag(X_rand_train)), axis=1)
X_rand_test = np.exp(-1j * X_test@W)
X_rand_test = np.concatenate((np.real(X_rand_test), np.imag(X_rand_test)), axis=1)
X_rand_test_ood = np.exp(-1j * X_test_ood@W)
X_rand_test_ood = np.concatenate((np.real(X_rand_test_ood), np.imag(X_rand_test_ood)), axis=1)


nrep = 100
n = 64


lam_list = np.arange(-100, 100)


np.random.seed(42)
n_remove = [0,1,2,3,4]

mse_in = np.zeros((lam_list.size, nrep))
mse_out = np.zeros((lam_list.size, nrep, len(n_remove)))

mse_out_per_class = np.zeros((lam_list.size, nrep, len(np.unique(y_test_ood))))
mineig = np.zeros(nrep)

for r in tqdm(range(nrep)):
    train = np.random.choice(X_rand_train.shape[0], size=n, replace=False)

    
    _X_rand_train = X_rand_train[train,:]
    
    muX = np.mean(_X_rand_train, axis=0)
    muY = np.mean(y_train[train])

    [U,s,V] = np.linalg.svd((_X_rand_train-muX), full_matrices = False)
    V = V.T
    
    # Remove near-zero singular value due to centering
    # This eliminates a little noisy MSE spike at lambda=0
    s = s[:-1]
    U = U[:,:-1]
    V = V[:,:-1]
    
    SVD_per = []
    for k in range(10):
        [_U,_s,_V] = np.linalg.svd((_X_rand_train-muX)[y_train[train]==k], full_matrices = False)
        _V = _V.T
        _s = _s[:-1]
        _U = _U[:,:-1]
        _V = _V[:,:-1]
        SVD_per.append([_U,_s,_V])
    
    mineig[r] = np.min(s)**2

    # ridge estimators with various lambdas
        
    def comp(l):
        _mse_out = np.zeros((len(n_remove)))
        _mse_out_per_class = np.zeros((len(np.unique(y_test_ood))))
        
        beta = V @ np.diag(s/(s**2 + l)) @ U.T @ (y_train[train]-muY)

        err = ((X_rand_test_ood-muX) @ beta - (y_test_ood-muY))**2
        for j in range(len(n_remove)):            
            _mse_out[j] = np.mean(err[~((y_test_ood>4-j)&(y_test_ood<=4))])
            
        for j in np.unique(y_test_ood):
            _U, _s, _V = SVD_per[j]
            beta_per = _V @ np.diag(_s/(_s**2 + l)) @ _U.T @ (y_train[train]-muY)[y_train[train]==j]

            _err = ((X_rand_test_ood-muX)[(y_test_ood==j)] @ beta_per 
                   - (y_test_ood-muY)[(y_test_ood==j)])**2
            _mse_out_per_class[j] = np.mean(err[(y_test_ood==j)]) - np.mean(_err)
        return _mse_out, _mse_out_per_class

    
    with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:
        res = parallel(
            delayed(comp)(l) for i,l in tqdm(enumerate(lam_list), desc='i')
        )
        res = list(zip(*res))
        
        mse_out[:,r,:] = np.array(res[0])
        mse_out_per_class[:,r,:] = np.array(res[1])
        

        
path_result = 'result/ex2/'
os.makedirs(path_result, exist_ok=True)

filename = path_result+'res_out.pkl'
with open(filename, 'wb') as f:
    np.save(f, mse_out)

filename = path_result+'res_out_per_class.pkl'
with open(filename, 'wb') as f:
    np.save(f, mse_out_per_class)
    
filename = path_result+'res_mineig.pkl'
df = pd.DataFrame(mineig, index=np.arange(nrep))
df.to_pickle(filename, compression='gzip')



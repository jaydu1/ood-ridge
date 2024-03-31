import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import fetch_openml

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
lam_list = np.arange(-100, 100)
n_remove = [0,1,2,3,4]

path_result = 'result/ex3/'
os.makedirs(path_result, exist_ok=True)
    
for n in np.arange(25,210,25):

    np.random.seed(42)


    mse_in = np.zeros((lam_list.size, nrep))
    mse_out = np.zeros((lam_list.size, nrep, len(n_remove)))
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

        mineig[r] = np.min(s)**2

        # ridge estimators with various lambdas    
        for i,l in enumerate(lam_list):
            beta = V @ np.diag(s/(s**2 + l)) @ U.T @ (y_train[train]-muY)

            err = ((X_rand_test_ood-muX) @ beta - (y_test_ood-muY))**2
            for j in range(len(n_remove)):            
                mse_out[i,r,j] = np.mean(err[~((y_test_ood>4-j)&(y_test_ood<=4))])

    filename = path_result+'res_out_{}.pkl'.format(n)
    with open(filename, 'wb') as f:
        np.save(f, mse_out)

    filename = path_result+'res_mineig_{}.pkl'.format(n)
    df = pd.DataFrame(mineig, index=np.arange(nrep))
    df.to_pickle(filename, compression='gzip')



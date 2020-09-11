import numpy as np
import os
import urllib.request
from scipy.io import loadmat



# 1D
name = "synthetic_data_1D"
f = lambda x: np.sin(np.pi * x)
fnametr = name+'_train.csv'
if not os.path.exists(fnametr):
    ndtr = 200
    xtr = np.random.rand(ndtr, 1) - 0.75
    ytr = f(xtr)
    np.savetxt(fnametr, np.concatenate((xtr, ytr), axis=1), delimiter=',')
    print('Generated Data ' + fnametr)
else:
    print('Data existed ' + fnametr)

fnamete = name+'_test.csv'
if not os.path.exists(fnamete):
    nte = 200
    xte = np.linspace(-5, 5, nte).reshape(-1, 1)
    yte = f(xte)
    np.savetxt(fnamete, np.concatenate((xte, yte), axis=1), delimiter=',')
    print('Generated Data ' + fnamete)
else:
    print('Data existed ' + fnamete)

# 2D Gaussian
name = "synthetic_data_2D_gaussian"
f = lambda x: 0.1 * x[:, :1] ** 3 + 0.5 * x[:, 1:] ** 2

fnametr = name+'_train.csv'
if not os.path.exists(fnametr):
    ndtr = 500
    xtr1 = np.random.multivariate_normal([0, 0], [[0.01, 0], [0, 0.05]], ndtr)
    xtr2 = np.random.multivariate_normal([1, 0], [[0.01, 0], [0, 0.05]], ndtr)
    xtr = np.concatenate((xtr1, xtr2), axis=0)
    ytr = f(xtr)
    np.savetxt(fnametr, np.concatenate((xtr, ytr), axis=1), delimiter=',')
    print('Generated Data ' + fnametr)
else:
    print('Data existed ' + fnametr)

fnamete = name + '_test.csv'
if not os.path.exists(fnamete):
    nte = 1000
    ndte = np.sqrt(nte).astype(int)
    xte1, xte2 = np.meshgrid(np.linspace(-1, 2, ndte),
                             np.linspace(-1, 2, ndte))
    xte = np.concatenate((xte1.reshape(-1, 1), xte2.reshape(-1, 1)), axis=1)
    yte = f(xte)
    np.savetxt(fnamete, np.concatenate((xte, yte), axis=1), delimiter=',')

    print('Generated Data ' + fnamete)
else:
    print('Data existed ' + fnamete)



# 2D Square
name = "synthetic_data_2D_square"
f = lambda x: 0.1 * x[:, :1] ** 3 + 0.5 * x[:, 1:] ** 2

fnametr = name+'_train.csv'
if not os.path.exists(fnametr):
    ndtr = 20
    xtr1 = np.concatenate((np.linspace(0, 1, ndtr), np.ones(ndtr),
                           np.linspace(1, 0, ndtr), np.zeros(ndtr)), axis=0)
    xtr2 = np.concatenate((np.zeros(ndtr), np.linspace(0, 1, ndtr),
                           np.ones(ndtr), np.linspace(1, 0, ndtr)), axis=0)
    xtr = np.concatenate((xtr1.reshape(-1, 1), xtr2.reshape(-1, 1)), axis=1)
    ytr = f(xtr)
    np.savetxt(fnametr, np.concatenate((xtr, ytr), axis=1), delimiter=',')
    print('Generated Data ' + fnametr)
else:
    print('Data existed ' + fnametr)

fnamete = name + '_test.csv'
if not os.path.exists(fnamete):
    nte = 1000
    ndte = np.sqrt(nte).astype(int)
    xte1, xte2 = np.meshgrid(np.linspace(-1, 2, ndte),
                             np.linspace(-1, 2, ndte))
    xte = np.concatenate((xte1.reshape(-1, 1), xte2.reshape(-1, 1)), axis=1)
    yte = f(xte)
    np.savetxt(fnamete, np.concatenate((xte, yte), axis=1), delimiter=',')
    print('Generated Data ' + fnamete)
else:
    print('Data existed ' + fnamete)

# Sarcos Data
filename = "sarcos_train.csv"
if not os.path.exists(filename):
    urllib.request.urlretrieve("http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat", "sarcos_inv.mat")
    train_data = loadmat('sarcos_inv.mat')['sarcos_inv'].astype(np.float32)
    np.savetxt(filename, train_data, delimiter=',')
    os.remove("sarcos_inv.mat")
    print('Generated Data ' + filename)
else:
    print('Data existed ' + filename)

filename = "sarcos_test.csv"
if not os.path.exists(filename):
    matfile = "sarcos_inv_test.mat"
    urllib.request.urlretrieve("http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat", matfile)
    train_data = loadmat(matfile)['sarcos_inv_test'].astype(np.float32)
    np.savetxt(filename, train_data, delimiter=',')
    os.remove(matfile)
    print('Generated Data ' + filename)
else:
    print('Data existed ' + filename)
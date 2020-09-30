import numpy as np
import os
import urllib.request
from scipy.io import loadmat

# 1D
name = "synthetic_data_1D_centered"
f = lambda x: np.sin(np.pi * x)
fnametr = name + '_train.csv'
if not os.path.exists(fnametr):
    np.random.seed(1)
    ndtr = 200
    xtr = 4 * np.random.rand(ndtr, 1) - 1
    ytr = f(xtr)
    np.savetxt(fnametr, np.concatenate((xtr, ytr), axis=1), delimiter=',')
    print('Generated Data ' + fnametr)
else:
    print('Data existed ' + fnametr)

fnamete = name + '_test.csv'
if not os.path.exists(fnamete):
    np.random.seed(1)
    nte = 200
    xte = np.linspace(-4, 4, nte).reshape(-1, 1)
    yte = f(xte)
    np.savetxt(fnamete, np.concatenate((xte, yte), axis=1), delimiter=',')
    print('Generated Data ' + fnamete)
else:
    print('Data existed ' + fnamete)

# 1D split
name = "synthetic_data_1D_split"
f = lambda x: np.sin(np.pi * x)
fnametr = name + '_train.csv'
if not os.path.exists(fnametr):
    np.random.seed(1)
    ndtr = 100
    xtr = np.concatenate((np.random.rand(ndtr, 1) - 2,
                          np.random.rand(ndtr, 1) + 1))
    ytr = f(xtr)
    np.savetxt(fnametr, np.concatenate((xtr, ytr), axis=1), delimiter=',')
    print('Generated Data ' + fnametr)
else:
    print('Data existed ' + fnametr)

fnamete = name + '_test.csv'
if not os.path.exists(fnamete):
    np.random.seed(1)
    nte = 200
    xte = np.linspace(-4, 4, nte).reshape(-1, 1)
    yte = f(xte)
    np.savetxt(fnamete, np.concatenate((xte, yte), axis=1), delimiter=',')
    print('Generated Data ' + fnamete)
else:
    print('Data existed ' + fnamete)

# 2D Gaussian
name = "synthetic_data_2D_gaussian"
eps = 1e-9 * np.random.rand(1)
f = lambda x: np.sin(5 * x[:, :1]) / (5 * x[:, :1] + eps) + x[:, 1:] ** 2

fnametr = name + '_train.csv'
if not os.path.exists(fnametr):
    np.random.seed(1)
    ndtr = 500
    xtr1 = np.random.multivariate_normal([-1, 0], [[0.02, 0], [0, 0.1]], ndtr)
    xtr2 = np.random.multivariate_normal([1, 0], [[0.02, 0], [0, 0.1]], ndtr)
    xtr = np.concatenate((xtr1, xtr2), axis=0)
    ytr = f(xtr)
    np.savetxt(fnametr, np.concatenate((xtr, ytr), axis=1), delimiter=',')
    print('Generated Data ' + fnametr)
else:
    print('Data existed ' + fnametr)

fnamete = name + '_test.csv'
if not os.path.exists(fnamete):
    np.random.seed(1)
    nte = 1000
    ndte = np.sqrt(nte).astype(int)
    xte1, xte2 = np.meshgrid(np.linspace(-2, 2, ndte),
                             np.linspace(-1, 1, ndte))
    xte = np.concatenate((xte1.reshape(-1, 1), xte2.reshape(-1, 1)), axis=1)
    yte = f(xte)
    np.savetxt(fnamete, np.concatenate((xte, yte), axis=1), delimiter=',')

    print('Generated Data ' + fnamete)
else:
    print('Data existed ' + fnamete)

# 2D Square
name = "synthetic_data_2D_square"
f = lambda x: np.sin(5 * x[:, :1]) / (5 * x[:, :1] + eps) + x[:, 1:] ** 2

fnametr = name + '_train.csv'
if not os.path.exists(fnametr):
    np.random.seed(1)
    ndtr = 20
    xtr1 = np.concatenate((np.linspace(-1, 1, ndtr), 1 * np.ones(ndtr),
                           np.linspace(1, -1, ndtr), -1*np.ones(ndtr)), axis=0)
    xtr2 = np.concatenate((-1*np.ones(ndtr), np.linspace(-1, 1, ndtr),
                           1 * np.ones(ndtr), np.linspace(1, -1, ndtr)), axis=0)
    xtr = np.concatenate((xtr1.reshape(-1, 1), xtr2.reshape(-1, 1)), axis=1)
    ytr = f(xtr)
    np.savetxt(fnametr, np.concatenate((xtr, ytr), axis=1), delimiter=',')
    print('Generated Data ' + fnametr)
else:
    print('Data existed ' + fnametr)

fnamete = name + '_test.csv'
if not os.path.exists(fnamete):
    np.random.seed(1)
    nte = 1000
    ndte = np.sqrt(nte).astype(int)
    xte1, xte2 = np.meshgrid(np.linspace(-2, 2, ndte),
                             np.linspace(-2, 2, ndte))
    xte = np.concatenate((xte1.reshape(-1, 1), xte2.reshape(-1, 1)), axis=1)
    yte = f(xte)
    np.savetxt(fnamete, np.concatenate((xte, yte), axis=1), delimiter=',')
    print('Generated Data ' + fnamete)
else:
    print('Data existed ' + fnamete)

# PMSM temperature
name = "pmsm_temperature"
filetr = name + '_train.csv'
filete = name + '_test.csv'
infile = 'pmsm_temperature_data.csv'
if not os.path.exists(filete) or not os.path.exists(filetr):
    if os.path.exists(infile):
        np.random.seed(1)
        ntr, nte = 5000, 1000
        n = ntr + nte
        data = np.genfromtxt(infile, delimiter=',', skip_header=1)[:n, :9]
        idxte = np.isin(np.arange(n), np.random.choice(n, nte, replace=False))
        idxtr = np.invert(idxte)
        np.savetxt(filetr, data[idxtr, :], delimiter=',')
        print('Generated Data ' + filetr)
        np.savetxt(filete, data[idxte, :], delimiter=',')
        print('Generated Data ' + filete)
    else:
        print(
            'please download' + infile + ' from https://www.kaggle.com/wkirgsn/electric-motor-temperature')
else:
    print('Data existed ' + filetr)
    print('Data existed ' + filete)

# Sarcos Data
filename = "sarcos_train.csv"
infile = "sarcos_inv_train.mat"

if not os.path.exists(filename):
    np.random.seed(1)
    urllib.request.urlretrieve(
        "http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat", infile)
    data = loadmat(infile)['sarcos_inv'].astype(np.float32)
    n = data.shape[0]
    ntr = 10000
    idx = np.random.choice(n, ntr, replace=False)
    np.savetxt(filename, data[idx, :22], delimiter=',')
    os.remove(infile)
    print('Generated Data ' + filename)
else:
    print('Data existed ' + filename)

filename = "sarcos_test.csv"
infile = "sarcos_inv_test.mat"
if not os.path.exists(filename):
    np.random.seed(1)
    urllib.request.urlretrieve(
        "http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat", infile)
    data = loadmat(infile)['sarcos_inv_test'].astype(np.float32)
    n = data.shape[0]
    nte = 2000
    idx = np.random.choice(n, nte, replace=False)
    np.savetxt(filename, data[idx, :22], delimiter=',')
    os.remove(infile)
    print('Generated Data ' + filename)
else:
    print('Data existed ' + filename)

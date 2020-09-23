from src import run
import numpy as np
import os
def test_main():
    name = "smoke"
    fnametr = '../data/' + name + '_train.csv'
    fnamete = '../data/' + name + '_test.csv'

    dx, dy = 2, 1
    ntr, nte = 10, 5
    xtr, ytr = np.random.randn(ntr, dx), np.random.randn(ntr, dy)
    xte, yte = np.random.randn(nte, dx), np.random.randn(nte, dy)
    np.savetxt(fnametr, np.concatenate((xtr, ytr), axis=1), delimiter=',')
    np.savetxt(fnamete, np.concatenate((xte, yte), axis=1), delimiter=',')

    run.main(True)

    os.remove(fnametr)
    os.remove(fnamete)
    os.remove('../results/smoke.txt')


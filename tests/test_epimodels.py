import numpy as np
dx, dy = 2, 1
ntr, nte = 10, 5
xtr, ytr = np.random.randn(ntr, dx), np.random.randn(ntr, dy)
xte, yte = np.random.randn(nte, dx), np.random.randn(nte, dy)


def test_gpmodel():
    from src.gpmodel import GPmodel
    model = GPmodel(dx = dx, dy = dy)
    model.train(xtr, ytr, display_progress = False)
    model.evaluate(xte, yte, xtr, ytr)

def test_bnn():
    from src.bnn import BNN
    model = BNN(dx = dx, dy = dy)
    model.train(xtr, ytr, display_progress = False)
    model.evaluate(xte, yte, xtr, ytr)

def test_dropout():
    from src.dropout import Dropout
    model = Dropout(dx = dx, dy = dy)
    model.train(xtr, ytr, display_progress = False)
    model.evaluate(xte, yte, xtr, ytr)

def test_negsep():
    from src.negsep import Negsep
    model = Negsep(dx = dx, dy = dy)
    model.train(xtr, ytr, display_progress = False)
    model.evaluate(xte, yte, xtr, ytr)



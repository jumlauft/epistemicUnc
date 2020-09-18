import numpy as np
from utils import eval_discounted_rmse


class EpiModel:
    def __init__(self, DX=1, DY=1):
        self.DX = DX
        self.DY = DY

    def add_data(self, xtr, ytr):
        """ Adds new training data points to the  model
        Args:
            xtr: input of data to be added
            ytr: output of data to be added
        """
        if not hasattr(self, 'Xtr'):
            self.Xtr = xtr
            self.Ytr = ytr
        else:
            self.Xtr = np.concatenate((self.Xtr, xtr), axis=0)
            self.Ytr = np.concatenate((self.Ytr, ytr), axis=0)


    def evaluate(self,xte,yte,xtr,ytr):
        ypredte, epite = self.predict(xte)
        ypredtr, epitr = self.predict(xtr)
        return eval_discounted_rmse(yte, ypredte, epite, ytr, ypredtr, epitr)


    def compare(self,xte,model):
        _, epi = self.predict(xte)
        _, epi_ref = model.predict(xte)
        return np.sqrt(((epi - epi_ref) ** 2).mean())
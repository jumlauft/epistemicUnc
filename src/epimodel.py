import numpy as np
from src.utils import eval_discounted_mse
from scipy.optimize import minimize

class EpiModel:
    def __init__(self, dx=1, dy=1):
        """ Parent class for different models predicting epistemic uncertainty

        Args:
            dx: input dimensions
            dy: output dimensions
        """
        self.DX = dx
        self.DY = dy

    def evaluate(self, xte, yte, xtr, ytr, min_alpha=True):
        """ Evaluates performance of epimodel at specified test /trainng points
        Args:
            xte: test inputs (dx x nte)
            yte: test outputs (dy x nte)
            xtr: trainning inputs (dx x ntr)
            ytr: training outputs (dy x ntr)

        Returns:
            dict containing different performance measures
        """
        ypredte, epite = self.predict(xte)
        ypredtr, epitr = self.predict(xtr)
        if min_alpha:
            def fun(a):
                r = eval_discounted_mse(yte, ypredte, a*epite, ytr, ypredtr, a*epitr)
                return r['mse_discounted_test']
            opt = minimize(fun,0.5*np.ones(1),bounds=((0,1),))
            a = opt.x
        else:
            a = 1
        return eval_discounted_mse(yte, ypredte, a * epite, ytr, ypredtr, a * epitr)

    def compare(self, xte, model):
        """ Compare epistemic uncertainty prediction to alternative model

        Args:
            xte: test inputs
            model: other model derived from EpiModel

        Returns:
            rmse difference in epi
        """
        _, epi = self.predict(xte)
        _, epi_ref = model.predict(xte)
        return np.sqrt(((epi - epi_ref) ** 2).mean())

    def get_x_epi(self):
        """ Placeholder: Returns epi inputs if available """
        return None

    def get_y_epi(self):
        """ Placeholder: Returns epi outputs if available """

        return None

import numpy as np
from src.utils import eval_discounted_mse


class EpiModel:
    def __init__(self, dx=1, dy=1):
        self.DX = dx
        self.DY = dy

    def evaluate(self, xte, yte, xtr, ytr):
        ypredte, epite = self.predict(xte)
        ypredtr, epitr = self.predict(xtr)
        return eval_discounted_mse(yte, ypredte, epite, ytr, ypredtr, epitr)

    def predict(self,x):
        n = x.shape[0]
        return np.zeros((n,self.DY)), np.ones((n,))

    def compare(self, xte, model):
        _, epi = self.predict(xte)
        _, epi_ref = model.predict(xte)
        return np.sqrt(((epi - epi_ref) ** 2).mean())

    def get_x_epi(self):
        return None

    def get_y_epi(self):
        return None

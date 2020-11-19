import GPy
from src.epimodel import EpiModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class GPmodel(EpiModel):
    def __init__(self, ARD=True, LENGTHSCALE=1, **kwargs):
        """ Wrapper for GP model
        Args:
            dx (int): input dimension
            dy (int): output dimension
            ARD (bool): automatic relevant determination
            LENGTHSCALE (float): lengthscale
        Attributes:
            kernel: kernel of GP
            GP: GP model from GPy

        """
        super().__init__(**kwargs)

        self.kernel = GPy.kern.RBF(input_dim=self.DX, ARD=ARD,
                                   lengthscale=LENGTHSCALE)
        self.GP = None

    def train(self, xtr, ytr, display_progress=False):
        """ Trains the GP
            likelihood maximization
        """
        self.GP = GPy.models.GPRegression(xtr, ytr, self.kernel)

        self.GP.optimize(messages=display_progress)

    def predict(self, x):
        """ Predicts outputs of the GP model for the given input x

        Args:
            x: input

        Returns:
            mean, epistemic uncertainty (scaled to [0,1])
        """
        (ypred, yvar) = self.GP.predict_noiseless(x)
        return ypred, MinMaxScaler().fit_transform(np.sqrt(yvar))

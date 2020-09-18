import numpy as np
import GPy
from epimodel import EpiModel
from sklearn.preprocessing import MinMaxScaler


class GPmodel(EpiModel):
    def __init__(self, ARD = True, LENGTHSCALE = 0.5 ,**kwargs):
        """ GP model
        Args:
            dx (int): input dimension
            dy (int): output dimension

        Attributes:
            DX (int): input dimension
            DY (int): output dimension
            _scaler (sklearn scaler): scaler for data

        """
        super().__init__(**kwargs)

        self.kernel = GPy.kern.RBF(input_dim=self.DX, ARD = ARD,
                              lengthscale=LENGTHSCALE)
        self.GP = None


    def train(self, xtr, ytr):
        """ Trains the neural network based on the current data

        Training iterates between training the disturbance output and the
        epistemic uncertainty output

        """
        self.GP = GPy.models.GPRegression(xtr, ytr, self.kernel)

        self.GP.optimize(messages=True)

    def predict(self, x):
        """ Predicts outputs of the NN model for the given input x

        Args:
            x: input

        Returns:
            mean, aleatoric uncertainty, epistemic uncertainty
        """
        (ypred, epi) = self.GP.predict_noiseless(x)
        return ypred, MinMaxScaler().fit_transform(epi)


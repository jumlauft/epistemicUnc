import numpy as np
import GPy


class GPmodel:
    def __init__(self, dx, dy):
        """ Online disturbance model to differentiate types of uncertainties

        Args:
            dx (int): input dimension
            dy (int): output dimension
            input_lb (list): length of DX list of minimal inputs
            input_up (list): length of DX list of maximal outputs

        Attributes:
            DX (int): input dimension
            DY (int): output dimension
            INPUT_LB (list): length of DX list of minimal inputs
            INPUT_UB (list): length of DX list of maximal outputs
            x_epi (numpy array): input locations where no data is available
            y_epi (numpy array): output indicating high/low uncertainty
            _scaler (sklearn scaler): scaler for data
            loss (list): loss over training epochs
            _train_counter (int): counts number of added points until retraining

            TRAIN_EPOCHS (int): Number of training epochs per iteration
            TRAIN_ITER (int): Number of training iterations
            N_HIDDEN (int): Number of hidden neurons per layer
            LEARNING_RATE (float): step size of RMSprop optimizer
            MOMENTUM (float): momentum of RMSprop optimizer
            SCALE_OFFSET (float): numerical stability for variance predictions
            MIN_ADD_DATA_RATE (float): lower bounds the acceptance probability for
                                        incoming data points
            N_EPI (int): number of additional data points stored for epsistemic
            TRAIN_LIM (int): upper bound for _train_counter (triggers retraining)


        """
        self.DX = dx
        self.DY = dy
        self.kernel = GPy.kern.RBF(input_dim=self.DX, ARD=True,
                              lengthscale=0.1)
        self.GP = None

    def _generate_rand_epi(self, n):
        """ Generates random input locations for epistemic uncertainty measure

        Uniformly distributes data points across the input space defined by
        INPUT_LB and INPUT_UB


        Args:
            n (int): Number of points to be generated

        Returns:
            [n, DX] numpy array

        """
        lim = np.array([self.INPUT_LB, self.INPUT_UB])
        return (lim[1, :] - lim[0, :]) * np.random.rand(n, self.DX) + lim[0, :]


    def train(self):
        """ Trains the neural network based on the current data

        Training iterates between training the disturbance output and the
        epistemic uncertainty output

        """
        # self.GP.optimize(messages=True)

    def predict(self, x):
        """ Predicts outputs of the NN model for the given input x

        Args:
            x: input

        Returns:
            mean, aleatoric uncertainty, epistemic uncertainty
        """
        (ypred, epi) = self.GP.predict_noiseless(x)
        return ypred, epi

    def predict_mean(self, x):
        """ Predicts mean outputs of the NN model for the given input x

        Args:
            x: input

        Returns:
            mean prediction
        """
        (mean, _) = self.GP.predict_noiseless(x)
        return mean.flatten()

    def predict_epistemic(self, x):
        """ Predicts epistemic uncertainty of the NN model for the given input x

        Args:
            x: input

        Returns:
            epistemic uncertainty prediction
        """
        (_, epi) = self.GP.predict_noiseless(x)

        return epi.flatten()

    def add_data(self, xtr, ytr):
        """ Adds new training data points to the disturbance model

        Selects data to be added and triggers retraining if necessary

        Args:
            xtr: input of data to be added
            ytr: output of data to be added
        """

        if not hasattr(self, 'Xtr'):
            # self.update_xy_epi(xtr)
            self.Xtr = xtr
            self.Ytr = ytr
        else:
            self.Xtr = np.concatenate((self.Xtr, xtr), axis=0)
            self.Ytr = np.concatenate((self.Ytr, ytr), axis=0)

        self.GP = GPy.models.GPRegression(self.Xtr,self.Ytr, self.kernel)

    def epi_accuracy(self,xte,yte):
        ypred, epi = self.predict(xte)
        return (((ypred-yte)**2)*(1-epi)).mean()*epi.mean()
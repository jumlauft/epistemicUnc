import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
from src.epimodel import EpiModel


class Negsep(EpiModel):
    def __init__(self, R_EPI=1, N_EPI=2, TRAIN_EPOCHS=5, TRAIN_ITER=2,
                 N_HIDDEN=10, LEARNING_RATE=0.01, **kwargs):
        """ Online disturbance model to differentiate types of uncertainties

        Args:
            dx (int): input dimension
            dy (int): output dimension


        Attributes:
            DX (int): input dimension
            DY (int): output dimension
            x_epi ((nepi,dx) np array): input locations where no data is available
            y_epi ((nepi,) np array): output indicating high/low uncertainty
            _scaler (sklearn scaler): scaler for data
            loss (list): loss over training epochs

            TRAIN_EPOCHS (int): Number of training epochs per iteration
            TRAIN_ITER (int): Number of training iterations
            N_HIDDEN (int): Number of hidden neurons per layer
            LEARNING_RATE (float): step size of RMSprop optimizer
            N_EPI (int): number of additional data points stored for epsistemic


        """
        super().__init__(**kwargs)
        self.R_EPI = R_EPI
        self.N_EPI = N_EPI
        self.TRAIN_EPOCHS = TRAIN_EPOCHS
        self.TRAIN_ITER = TRAIN_ITER
        self._scaler = StandardScaler()

        tf.random.set_seed(0)
        inp = Input(shape=(self.DX,))
        hidden = Dense(N_HIDDEN, activation="relu")(inp)
        hidden = Dense(N_HIDDEN, activation="relu")(hidden)

        meanlay = Dense(self.DY)(hidden)
        epilay = Dense(1, activation='sigmoid')(hidden)

        self.model_epi = Model(inputs=inp, outputs=epilay)
        self.model_epi.compile(
            optimizer=tf.optimizers.RMSprop(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy')

        self.model_mean = Model(inputs=inp, outputs=meanlay)
        self.model_mean.compile(
            optimizer=tf.optimizers.RMSprop(learning_rate=LEARNING_RATE),
            loss='mse')

        self.model_all = Model(inputs=inp, outputs=[meanlay, epilay])

    def predict(self, x):
        """ Predicts outputs of the NN model for the given input x

        Args:
            x: input

        Returns:
            mean, aleatoric uncertainty, epistemic uncertainty
        """
        ypred, epi = self.model_all.predict(self._scaler.transform(x))
        return ypred, MinMaxScaler().fit_transform(epi)

    def train(self, xtr, ytr, display_progress=False):
        """ Adds new training data points to the disturbance model

        Selects data to be added and triggers retraining if necessary

        Args:
            xtr: input of data to be added
            ytr: output of data to be added
            display_progress: boolean
        """

        xtra = self._scaler.fit_transform(xtr)
        self.x_epi, self.y_epi = self._generate_xy_epi(xtra)

        cw = compute_class_weight('balanced', np.unique(self.y_epi),
                                  self.y_epi.flatten())
        loss = []
        for i in range(self.TRAIN_ITER):
            hist_epi = self.model_epi.fit(self.x_epi, self.y_epi,
                                          class_weight=cw,
                                          epochs=self.TRAIN_EPOCHS,
                                          verbose=int(display_progress))
            loss.extend(hist_epi.history['loss'])
            hist = self.model_mean.fit(xtra, ytr,
                                       epochs=self.TRAIN_EPOCHS, verbose=0)
            loss.extend(hist.history['loss'])
        return loss

    def _generate_xy_epi(self, xtra):
        """ Generates artificial data points for epistemic uncertainty estimate

        Args:
            xtra: scaled input training data points

        """
        ntr = xtra.shape[0]


        # Generate uncertain points (on GPU if available)
        cov = self.R_EPI
        Nepi = self.N_EPI * self.DX

        if len(tf.config.list_physical_devices('GPU')) == 0:
            from scipy.spatial.distance import cdist

            print('Sampling EPI points on CPU')
            cov_mat = cov * np.eye(self.DX)
            x_epilist = []
            distance = []
            for x in xtra:
                xepi = np.random.multivariate_normal(x, cov_mat, Nepi)
                x_epilist.append(xepi)
                distance.extend(cdist(xepi, xtra).min(axis=1))
            x_epi = np.concatenate(x_epilist, axis=0)
            d = np.array(distance)
        else:
            from numba import cuda
            from numba.cuda.random import create_xoroshiro128p_states, \
                xoroshiro128p_normal_float32
            print('Sampling EPI points on GPU')
            xtr = np.ascontiguousarray(xtra, dtype=np.float32)

            # cuda.select_device(1)
            @cuda.jit
            def generate_rand(rng_states, xtr, cov, Xepi, d):
                i = cuda.grid(1)
                ntr, dx = xtr.shape
                if i < ntr:
                    # Generate random points
                    for nepi in range(Nepi):
                        for dx in range(dx):
                            Xepi[i, dx, nepi] = xtr[i, dx] + \
                                                math.sqrt(float(cov)) * \
                                                xoroshiro128p_normal_float32(
                                                    rng_states, i)
                            # Compute distances
                    for nepi in range(Nepi):
                        smallest = 1e9
                        for nt in range(ntr):
                            dist = 0
                            for dx in range(dx):
                                dist += (Xepi[i, dx, nepi] - xtr[nt, dx]) ** 2
                            if dist < smallest:
                                smallest = dist
                        d[i, nepi] = smallest

            threads_per_block = 128
            blocks_per_grid = math.ceil(ntr / threads_per_block)
            rng_states = create_xoroshiro128p_states(ntr, seed=1)
            x_epi = np.zeros((ntr, self.DX, Nepi), dtype=np.float32)
            d = np.zeros((ntr, Nepi), dtype=np.float32)
            generate_rand[blocks_per_grid, threads_per_block](rng_states, xtr,
                                                              cov, x_epi, d)
            x_epi = x_epi.reshape(-1, self.DX)

        y_epi = np.ones((x_epi.shape[0], 1))
        idx = np.argpartition(d.reshape(-1), ntr)

        # replace by training data input and turn into certain points
        y_epi[idx[:ntr], :] = 0
        x_epi[idx[:ntr], :] = xtra
        return x_epi, y_epi

    def get_x_epi(self):
        return self._scaler.inverse_transform(self.x_epi)

    def get_y_epi(self):
        return self.y_epi

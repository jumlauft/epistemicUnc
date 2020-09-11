import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from numba import cuda 
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
import math


class NegSEp:
    TRAIN_EPOCHS = 1
    TRAIN_ITER = 3
    N_HIDDEN = 50
    LEARNING_RATE = 0.01
    MOMENTUM = 0.0001
    RADIUS_TR = 0.0001
    N_EPI = 1000

    def __init__(self, dx, dy, input_lb, input_up):
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
        self.INPUT_LB = input_lb
        self.INPUT_UB = input_up
        self._scaler = StandardScaler()
        self._train_counter = 0
        self.loss = []
        self.model_epi, self.model_mean, self.model_all = self._setup_nn


    @property
    def _setup_nn(self):
        """ Sets up the neural network structure for the disturbance model

        The neural network has three outputs
        - disturbance estimation
        - epsistemic uncertainty estimate

        Returns:
            model_epi: model to epistemic uncertainty prediction
            model_mean: model to mean output prediction
            model_all: model to all outputs at once

        """
        inp = Input(shape=(self.DX,))
        hidden = Dense(self.N_HIDDEN, activation="relu")(inp)
        hidden = Dense(self.N_HIDDEN, activation="relu")(hidden)

        meanlay = Dense(self.DY)(hidden)
        epilay = Dense(1, activation='sigmoid')(hidden)


        model_epi = Model(inputs=inp, outputs=epilay)
        model_epi.compile(
            optimizer=tf.optimizers.RMSprop(learning_rate=self.LEARNING_RATE),
            loss='binary_crossentropy')

        model_mean = Model(inputs=inp, outputs=meanlay)
        model_mean.compile(
            optimizer=tf.optimizers.RMSprop(learning_rate=self.LEARNING_RATE),
            loss='mse')

        model_all = Model(inputs=inp, outputs=[meanlay, epilay])
        return model_epi, model_mean, model_all

    def train(self):
        """ Trains the neural network based on the current data

        Training iterates between training the disturbance output and the
        epistemic uncertainty output

        """
        self._scaler.fit(self.x_epi)

        cw = compute_class_weight('balanced', np.unique(self.y_epi),
                                  self.y_epi.flatten())
        xepis = self._scaler.fit_transform(self.x_epi)
        xtrs = self._scaler.transform(self.Xtr)
        for i in range(self.TRAIN_ITER):
            # hist = self.model_out.fit(self.Xtr, self.Ytr, **kwargs)
            hist_epi = self.model_epi.fit(xepis, self.y_epi, class_weight=cw,
                                          epochs=self.TRAIN_EPOCHS, verbose=1)
            hist = self.model_mean.fit(xtrs, self.Ytr,
                                       epochs=self.TRAIN_EPOCHS, verbose=1)

            if np.isnan(hist_epi.history['loss']).any():
                print('detected Nan')
            self.loss = self.loss + hist.history['loss'] + hist_epi.history[
                'loss']
        return self.loss

    def predict(self, x):
        """ Predicts outputs of the NN model for the given input x

        Args:
            x: input

        Returns:
            mean, aleatoric uncertainty, epistemic uncertainty
        """
        mean, epi = self.model_all.predict(self._scaler.transform(x))
        return mean.flatten(), epi.flatten()


    def add_data(self, xtr, ytr):
        """ Adds new training data points to the disturbance model

        Selects data to be added and triggers retraining if necessary

        Args:
            xtr: input of data to be added
            ytr: output of data to be added
            epi_pred: epistemic uncertainty prediction at xtr
        """

        if not hasattr(self, 'Xtr'):
            # self.update_xy_epi(xtr)
            self.Xtr = xtr
            self.Ytr = ytr
        else:
            self.Xtr = np.concatenate((self.Xtr, xtr), axis=0)
            self.Ytr = np.concatenate((self.Ytr, ytr), axis=0)

        self.x_epi, self.y_epi = self._generate_xy_epi()

    def _generate_xy_epi(self):
        """ Generates artificial data points for epistemic uncertainty estimate

        """
        ntr = self.Xtr.shape[0]

        # ALTERNATIVE 1
        # distance = np.sum((self.x_epi.reshape(1,-1,self.DX)
        #               - self.Xtr.reshape(ntr,1,self.DX))**2, axis=2)
        # dis1fill = distance.min(axis = 0).reshape(-1,1)
        # RADIUS_TR = 0.0001
        # self.y_epi = (dis1fill > RADIUS_TR).astype(int)

        # ALTERNATIVE 2
        # Generate uncertain points
        # self.x_epi = self._generate_rand_epi(self.N_EPI + ntr)
        # self.y_epi = np.ones((self.N_EPI + ntr, 1))

        # ALTERNATIVE 3
        # Generate uncertain points
        cov = 0.2
        Nepi = 2 * self.DX
        
        if tf.config.list_physical_devices('GPU') == 0:
            print('Sampling EPI points on CPU')
            cov_mat = 0.2 * np.eye(self.DX)
            x_epilist = []
            distance = []
            for x in self.Xtr[:100,:]:
                xepi = np.random.multivariate_normal(x, cov_mat, Nepi)
                x_epilist.append(xepi)
                distance.extend(cdist(xepi,self.Xtr).min(axis=1))
            x_epi = np.concatenate(x_epilist, axis=0)
            d = np.array(distance)
        else:
            print('Sampling EPI points on GPU')
            Xtr = np.ascontiguousarray(self.Xtr, dtype = np.float32)
            # cuda.select_device(2)
            @cuda.jit
            def generate_rand(rng_states, Xtr, cov, Xepi, d):
                thread_id = cuda.grid(1)
                ntr, Dx = Xtr.shape
                if thread_id < ntr:
                    # Generate random points
                    for nepi in range(Nepi):
                        for dx in range(Dx):
                            Xepi[thread_id,dx,nepi] = Xtr[thread_id,dx] + \
                                        math.sqrt(cov)*xoroshiro128p_normal_float32(rng_states, thread_id) 
                    # Compute distances
                    for nepi in range(Nepi):
                        smallest = 1e9
                        for nt in range(ntr):
                            dist = 0
                            for dx in range(Dx):
                                dist += (Xepi[thread_id,dx,nepi] - Xtr[nt,dx])**2
                            if dist < smallest:
                                smallest = dist
                        d[thread_id,nepi] = smallest

            threads_per_block = 128
            blocks_per_grid = math.ceil(ntr / threads_per_block)
            rng_states = create_xoroshiro128p_states(ntr, seed=1)
            x_epi = np.zeros((ntr,self.DX,Nepi), dtype=np.float32)
            d = np.zeros((ntr,Nepi), dtype=np.float32)
            generate_rand[blocks_per_grid, threads_per_block](rng_states, Xtr, cov, x_epi,d)
            x_epi = x_epi.reshape(-1,self.DX)
        

        
        y_epi = np.ones((x_epi.shape[0], 1))
        idx = np.argpartition(d.reshape(-1), ntr)

        # find closest uncertain points
        # d = x_epi.reshape(1, -1, self.DX) - self.Xtr.reshape(-1, 1, self.DX)
        # distance = np.sum(d ** 2, axis=2)
        # idx = np.argpartition(distance.min(axis=0), ntr)



        # replace by training data input and turn into certain points
        y_epi[idx[:ntr], :] = 0
        x_epi[idx[:ntr], :] = self.Xtr
        return x_epi, y_epi

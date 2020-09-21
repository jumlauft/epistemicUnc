# -*- coding: utf-8 -*-
# Created with Python 3.6
"""
Bayesian fully-connected feedforward neural network
"""

import autograd.numpy as np
from autograd import grad
import pickle
import datetime
import copy
import keyboard
import gc


# Rectified Linear Unit
def ReLU(o):
    out = np.maximum(o,0)
    return out

# Trivial Linear Unit
def TrLU(o):
    return o

# Logistic Function 
def logistic(x):
    return 1/(1+np.exp(-x))



class BayesianNeuralNetwork:
    TRAIN_EPOCHS = 1000
    LEARNING_RATE = 0.001
    FEAT = 5
    W_V_LOG_INIT = -10
    W_Z_LOG_INIT = 10
    K = 25
    TH_Z = 1000
    CAP_Z = 0.1
    CAP_W = 1e-8

    def __init__(self, dx, dy):
        layer_sizes = [1 + 1 + dx, 10, 10, dy]

        # Bias and z assumed included in provided layer size
        self.layer_sizes = layer_sizes.copy()
        layer_sizes[-1] = layer_sizes[
                              -1] + 1  # This facilitates weights initialization
        # Initialize weights and their prior variance
        self.w_m = []
        self.w_v_log = []
        self.w_p_v = 1
        # Initialize random feature z's prior
        self.z_p_v = np.sqrt(self.FEAT)
        # Initialize additive output noise e
        self.e_m = np.zeros((1, self.layer_sizes[-1]))
        self.e_v = np.ones((1, self.layer_sizes[-1]))
        self.activations = []

        # Initialize all weights' dimensions (always subtract bias)
        # Initialize all but last layer's activation functions
        for i in range(0, len(layer_sizes) - 1):
            w_m = np.zeros(
                (layer_sizes[i], layer_sizes[i + 1] - 1)) + np.random.uniform(
                -0.05, 0.05, [layer_sizes[i], layer_sizes[i + 1] - 1])
            w_v_log = np.ones(
                (layer_sizes[i], layer_sizes[i + 1] - 1)) * self.W_V_LOG_INIT
            self.w_m.append(w_m)
            self.w_v_log.append(w_v_log)
            self.activations.append(ReLU)
        # Last layer activates trivially
        self.activations[-1] = TrLU
        layer_sizes[-1] = layer_sizes[
                              -1] - 1  # Undo weights initialization facilitation

        # Set up logistic conversion methods for w_v and z_v
        self.logistic_w = lambda w_v_log: [logistic(w_v_log[mesh]) for mesh
                                           in range(0, len(w_v_log))]
        self.logistic_z = logistic

        # Initialize standardization parameters
        self.x_mean = []
        self.x_std = []
        self.y_mean = []
        self.y_std = []

        # Initialize training parameters
        # Minibatch size
        self.S = 20
        # Alpha
        self.alpha = 0.5
    def standardize(self, x, y=[]):

        # Extract or use existing normal parameters to normalize samples
        if len(self.x_mean) == 0:
            self.x_mean = np.mean(x, axis=0)
            self.x_std = np.std(x, axis=0)
        x = (x - self.x_mean) / self.x_std
        # Extract or use existing normal parameters to normalize targets if there are any
        if len(y) != 0:
            if len(self.y_mean) == 0:
                self.y_mean = np.mean(y, axis=0)
                self.y_std = np.std(y, axis=0)
            y = (y - self.y_mean) / self.y_std
            return x, y
        else:
            return x

    def unstandardize(self, x, y):

        # Unstandardize samples and targets
        x = x * self.x_std + self.x_mean
        y = y * self.y_std + self.y_mean
        return x, y

    def build_minibatches(self, x, y, z_m, z_v):

        # Calculate how many minibatches we'll end up with
        no_mbs = np.floor(x.shape[0] / self.S).astype(np.int)
        # Generate matrices that hold the batches
        x_MBs = np.zeros((no_mbs, self.S, x.shape[1]))
        y_MBs = np.zeros((no_mbs, self.S, y.shape[1]))
        z_m_MBs = np.zeros((no_mbs, self.S, z_m.shape[1]))
        z_v_MBs = np.zeros((no_mbs, self.S, z_v.shape[1]))
        # Shape the batches and append
        for mb in range(0, no_mbs):
            x_MB = x[mb * self.S:(mb + 1) * self.S, :]
            y_MB = y[mb * self.S:(mb + 1) * self.S, :]
            z_m_MB = z_m[mb * self.S:(mb + 1) * self.S, :]
            z_v_MB = z_v[mb * self.S:(mb + 1) * self.S, :]
            x_MBs[mb] = x_MB
            y_MBs[mb] = y_MB
            z_m_MBs[mb] = z_m_MB
            z_v_MBs[mb] = z_v_MB
        return x_MBs, y_MBs, z_m_MBs, z_v_MBs

    def sample_NNs(self, w_m, w_v):
        # NNs: deterministic neural networks, zs: random input features, es: additive output noise.
        # Sample K deterministic neural networks with output noise
        NNs = []
        for k in range(0, self.K):
            # Generate weights
            w = []
            for mesh in range(0, len(w_m)):
                rnd = np.random.randn(w_m[mesh].shape[0], w_m[mesh].shape[1])
                w.append(w_m[mesh] + rnd * w_v[mesh])
            NN = DNN(w, self.activations)
            NNs.append(NN)
        return NNs

    def sample_zs(self, z_m, z_v, N):
        # Sample K sets of N random features
        zs = []
        for k in range(0, self.K):
            rnd = np.random.randn(N, z_m.shape[1])
            z = z_m + rnd * z_v
            zs.append(z)
        return zs

    def sample_es(self, e_v, N):
        # Generate additive output noise for K NNs and N NN-outputs
        rnd = np.random.randn(self.K, N, self.e_m.shape[1])
        es = self.e_m + rnd * e_v
        return es

    def compute_norm_factor(self, w_m, w_v, z_m, z_v):

        # Calculate normalization factor for the weights
        log_n_w = 0
        for mesh in range(0, len(w_m)):
            log_n_w = log_n_w + np.sum(
                0.5 * np.log(2 * np.pi * w_v[mesh]) + np.square(w_m[mesh]) /
                w_v[mesh])
        # And for the random features
        log_n_z = np.sum(0.5 * np.log(2 * np.pi * z_v) + np.square(z_m) / z_v)
        # Assemble total normalization factor
        log_n = log_n_w + log_n_z
        return log_n

    def compute_LL_factors(self, NNs, zs, w_m, w_v, z_m, z_v, N):

        # Calculate K likelihood factors for the weights
        f_ws = []
        f_zs = []
        for k in range(0, self.K):
            # For the weights
            f_w = 0
            for mesh in range(0, len(NNs[k].w)):
                # Split long expression into two
                f1_w = ((self.w_p_v * w_v[mesh]) / (
                            self.w_p_v - w_v[mesh])) * np.square(NNs[k].w[mesh])
                # f1_w = ((w_v[mesh] - self.w_p_v)/(2*self.w_p_v*w_v[mesh])) * np.square(NNs[k].w[mesh]) Depeweg's correction
                f2_w = (w_m[mesh] / w_v[mesh]) * NNs[k].w[mesh]
                f_w = f_w + (np.sum(f1_w) + np.sum(f2_w)) / N
            f_w = f_w * self.alpha
            f_w = np.exp(f_w)
            if f_w > 1e150:
                print('Warning: an f_w converges to inf.')
                self.errormsg.append('an f_w goes to inf.')
            f_ws.append(f_w)
            # For the random features z
            f1_z = ((self.z_p_v * z_v) / (self.z_p_v - z_v)) * np.square(zs[k])
            # f1_z = ((z_v-self.z_p_v)/(2*self.z_p_v*z_v)) * np.square(zs[k]) Depeweg's correction
            f2_z = (z_m / z_v) * zs[k]
            f_z = (f1_z + f2_z) * self.alpha
            f_z = np.exp(f_z)
            if (f_z > 1e150).any() == True:
                print('Warning: an f_z converges to inf.')
                self.errormsg.append('an f_z goes to inf.')
            f_zs.append(f_z)
        return f_ws, f_zs

    def compute_sample_LLs(self, NNs, zs, w_m, w_v, z_m, z_v, e_v, x, y, N):

        # Compute likelihood factors
        f_ws, f_zs = self.compute_LL_factors(NNs, zs, w_m, w_v, z_m, z_v, N)
        # Calculate likelihoods for every data-pair in the minibatch
        lls = []
        denom = 2 * e_v
        for k in range(0, self.K):
            # Append random features z to x
            x_z = np.concatenate((x, zs[k]), axis=1)
            out = NNs[k].execute(x_z)
            nom = np.square(y - out)
            ll = np.exp(-nom / denom) / (np.sqrt(2 * np.pi * e_v)) + 1e-10
            # Multiply multi-dimensional output if applicable
            ll = np.prod(ll, axis=1, keepdims=True)
            if (ll == 0).any() == True:
                print('Warning: A likelihood is zero.', np.argmin(ll))
                self.errormsg.append('one ll is zero.', np.argmin(ll))
            # Include alpha and divide by likelihood factors
            factored_ll = (ll ** self.alpha / (f_ws[k] * f_zs[k]))
            if (factored_ll == 0).any() == True:
                print('Warning: A factored likelihood is zero.')
                self.errormsg.append('one f_ll is zero.')
            lls.append(factored_ll)
        return lls

    def compute_BNN_LL(self, lls, no_mbs, N):

        # Compute the likelihood of the BNN parameters (second part energy function)
        ll_sums_per_sample = np.zeros((self.S, lls[0].shape[1]))
        for k in range(0, self.K):
            ll_sums_per_sample = ll_sums_per_sample + lls[k]
        log_ll_sums_per_sample = np.log(ll_sums_per_sample / self.K)
        ll_BNN = N / (self.alpha * self.S) * np.sum(log_ll_sums_per_sample)
        return ll_BNN

    def calculate_energy(self, tbo_pars, x, y, N):

        # Unpack parameters
        w_m = tbo_pars[0]
        w_v = self.logistic_w(tbo_pars[1])
        z_m = tbo_pars[2]
        z_v = self.logistic_z(tbo_pars[3])
        e_v = tbo_pars[4]
        # Generate deterministic neural networks and feature noise
        zs = self.sample_zs(z_m, z_v, self.S)
        NNs = self.sample_NNs(w_m, w_v)
        # Compute normalization factor of the approximating distribution
        log_n = self.compute_norm_factor(w_m, w_v, z_m, z_v)
        # Compute likelihoods for all data-pairs per sampled NN
        lls = self.compute_sample_LLs(NNs, zs, w_m, w_v, z_m, z_v, e_v, x, y, N)
        # Compute the average normalized likelihood of the BNN's parameters given the minibatch
        no_mbs = 1  # Adjust for MB training
        ll_BNN = self.compute_BNN_LL(lls, no_mbs, N)
        # Calculate the energy value
        energy = - log_n - ll_BNN
        return energy

    def adam_deluxe(self, tbo_pars, b1=0.9, b2=0.999, eps=10 ** -8):

        # Initialize m and v for every parameter
        m = []
        v = []
        elist = []
        tbo_pars_iter = []
        for par in tbo_pars:
            if type(par) == list:
                m_par = []
                v_par = []
                for mesh in par:
                    m_par.append(np.zeros((mesh.shape[0], mesh.shape[1])))
                    v_par.append(np.zeros((mesh.shape[0], mesh.shape[1])))
            else:
                m_par = np.zeros((par.shape[0], par.shape[1]))
                v_par = np.zeros((par.shape[0], par.shape[1]))
            m.append(m_par)
            v.append(v_par)

        # Iterate updates
        for i in range(self.TRAIN_EPOCHS):

            # Stop or pause training by pressing specified key
            if keyboard.is_pressed('ctrl+shift+s') == True:
                return tbo_pars, tbo_pars_iter
            elif keyboard.is_pressed('ctrl+shift+p') == True:
                input("Enter sth to resume training")

            # Save current params, calculate energy and stop training once energy is nan
            prior_pars = copy.deepcopy(tbo_pars)
            tbo_pars_iter.append(prior_pars)
            e = np.round(self.energy(tbo_pars), 4)
            if i % 100 == 0:
                print(i, 'Energy: ', e)
                elist.append(e)
            if np.isnan(e):
                return elist, tbo_pars, tbo_pars_iter
            # Get the gradients wrt parameters to-be-optimized
            grads = self.energy_grad(tbo_pars)
            # Optimize parameters one by one
            for par_n in range(0, len(tbo_pars)):
                if type(tbo_pars[par_n]) == list:
                    # Optimize weights mesh by mesh
                    for mesh in range(0, len(tbo_pars[par_n])):
                        m[par_n][mesh] = (1 - b1) * grads[par_n][mesh] + b1 * \
                                         m[par_n][mesh]
                        v[par_n][mesh] = (1 - b2) * (
                                    grads[par_n][mesh] ** 2) + b2 * v[par_n][
                                             mesh]
                        mhat = m[par_n][mesh] / (1 - b1 ** (i + 1))
                        vhat = v[par_n][mesh] / (1 - b2 ** (i + 1))
                        tbo_pars[par_n][mesh] = tbo_pars[par_n][
                                                    mesh] - self.LEARNING_RATE * mhat / (
                                                            np.sqrt(vhat) + eps)
                else:
                    m[par_n] = (1 - b1) * grads[par_n] + b1 * m[
                        par_n]  # First  moment estimate
                    v[par_n] = (1 - b2) * (grads[par_n] ** 2) + b2 * v[
                        par_n]  # Second moment estimate
                    mhat = m[par_n] / (1 - b1 ** (i + 1))  # Bias correction
                    vhat = v[par_n] / (1 - b2 ** (i + 1))
                    tbo_pars[par_n] = tbo_pars[
                                          par_n] - self.LEARNING_RATE * mhat / (
                                                  np.sqrt(vhat) + eps)

            # Check if any weight variance is too small (would result in infinite f_w)
            # tbo_pars = adam_validate_variances(tbo_pars)

            # Show statistics
            if i % 100 == 0:
                print('MSE:', np.round(self.MSE(), 6))
                adam_training_statistics(prior_pars, tbo_pars)

            # Collect garbage
            gc.collect()

        return elist, tbo_pars, tbo_pars_iter

    def train(self):

        self.errormsg = []
        N = self.Xtr.shape[0]
        self.S = N  # Adjust for MB training
        # Initialize array of random feature means and variances to fit
        z_m = np.zeros((N, 1))
        z_v_log = np.ones((N, 1)) * self.W_Z_LOG_INIT
        # Declare MSE
        self.MSE = lambda: np.mean(
            np.square(self.Ytr - np.mean(self.execute(self.Xtr), axis=0)))

        # Optimize parameters by minimizing the black-box alpha-divergence energy function
        # Declare energy function
        self.energy = lambda tbo_pars: self.calculate_energy(tbo_pars, self.Xtr, self.Ytr, N)
        # Declare gradient of energy wrt parameters to-be-optimized
        self.energy_grad = grad(self.energy, 0)
        # Wrap to-be-optimized parameters and optimize
        tbo_pars = [self.w_m, self.w_v_log, z_m, z_v_log, self.e_v]
        elist, tbo_pars, tbo_pars_iter = self.adam_deluxe(tbo_pars)
        # Reassign parameters
        self.w_m = tbo_pars[0]
        self.w_v_log = tbo_pars[1]
        self.e_v = tbo_pars[4]
        # Return course of parameter optimization
        return elist, tbo_pars_iter

    def execute(self, x):

        N = x.shape[0]
        # Generate K x N random features z
        z_m = np.zeros((N, 1))
        z_v = np.ones((N, 1)) * self.z_p_v
        # zs = self.sample_zs(z_m, z_v, N)
        zs = z_m + z_v * np.random.randn(self.K, N, z_m.shape[1])

        # Sample K deterministic neural networks
        w_v = self.logistic_w(self.w_v_log)
        NNs = self.sample_NNs(self.w_m, w_v)
        # Sample K x N times additive noise e
        es = self.sample_es(self.e_v, N)

        # Initialize output array
        out = np.zeros((self.K, N, self.layer_sizes[-1]))
        # Get perturbed outputs of K NNs
        for k in range(0, self.K):
            # Append features z to x, execute NNs, perturb with e
            x_z = np.concatenate((x, zs[k]), axis=1)
            out[k, :, :] = NNs[k].execute(x_z) + es[k, :, :]
        return out

    def add_data(self, xtr, ytr):
        """ Adds new training data points to the  model
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

    def execute_epistemic(self, x):

        N = x.shape[0]
        # Generate K x N random features z
        z_m = np.zeros((N, 1))
        z_v = np.ones((N, 1)) * self.z_p_v
        # zs = self.sample_zs(z_m, z_v, N)
        zs = z_m + z_v * np.random.randn(10 * self.K, N, z_m.shape[1])

        # Sample K deterministic neural networks
        w_v = self.logistic_w(self.w_v_log)
        NNs = self.sample_NNs(self.w_m, w_v)
        # Sample K x N times additive noise e
        es = self.sample_es(self.e_v, N)

        # Initialize output array
        out = np.zeros((self.K * 10, self.K, N, self.layer_sizes[-1]))
        # Get perturbed outputs of K NNs
        for kk in range(self.K * 10):
            for k in range(0, self.K):
                # Append features z to x, execute NNs, perturb with e
                x_z = np.concatenate((x, zs[kk]), axis=1)
                out[kk, k, :, :] = NNs[k].execute(x_z) + es[k, :, :]
        return out

    def predict(self, x):
        """ Predicts outputs of the NN model for the given input x

        Args:
            x: input

        Returns:
            mean, aleatoric uncertainty, epistemic uncertainty
        """
        y_test_epist = self.execute_epistemic(x)
        epi = y_test_epist.mean(axis = 0).std(axis = 0)
        mean = self.execute(x).mean(axis=0)
        return mean, epi
    def epi_accuracy(self,xte,yte):
        ypred, epi = self.predict(xte)
        iepi = 1/epi
        return (iepi*((ypred-yte)**2)).mean()/iepi.mean()
    def reparameterize(self, pars):
        self.w_m = pars[0]
        self.w_v_log = pars[1]
        self.e_v = pars[-1]

    def save(self, note=0):

        # Wrap NN parameters
        NN_params = []
        NN_params.append(self.layer_sizes)
        NN_params.append(self.activations)
        NN_params.append(self.z_p_v)
        NN_params.append(self.w_m)
        NN_params.append(self.w_v_log)
        NN_params.append(self.e_v)
        NN_params.append(self.x_mean)
        NN_params.append(self.x_std)
        NN_params.append(self.y_mean)
        NN_params.append(self.y_std)

        # Save them
        present = datetime.date.today().strftime('%Y%m%d')
        if note == 0:
            name = '_BNN_params.pkl'
        else:
            name = '_BNN_params_' + note + '.pkl'
        with open('03_Models/' + present + name, 'wb') as fhandle:
            pickle.dump(NN_params, fhandle)


# %%

# Deterministic neural network
class DNN:

    def __init__(self, w, acts):
        # Deterministic neural network with scalar weights
        self.w = w
        self.activations = acts

    def execute(self, x):
        # Append a bias to every input
        biasses = np.ones((x.shape[0], 1))
        layer_input = np.concatenate((x, biasses), axis=1)
        # For every computing layer
        for layer in range(0, len(self.w)):
            # Compute the layer's output as its neurons' activated sums of weighted inputs
            layer_output = self.activations[layer](
                np.dot(layer_input, self.w[layer]))
            # Assign the biassed output as input to the next layer
            layer_input = np.concatenate((layer_output, biasses), axis=1)
        # Input passed through the entire network yields the prediction y
        out = layer_output
        return out



# Load BNN
def load_BNN(fname):
    with open('03_Models/' + fname, 'rb') as fhandle:
        NN_params = pickle.load(fhandle)
        BNNET = BayesianNeuralNetwork(NN_params[0])
        BNNET.activations = NN_params[1]
        BNNET.z_p_v = NN_params[2]
        BNNET.w_m = NN_params[3]
        BNNET.w_v_log = NN_params[4]
        BNNET.e_v = NN_params[5]
        BNNET.x_mean = NN_params[6]
        BNNET.x_std = NN_params[7]
        BNNET.y_mean = NN_params[8]
        BNNET.y_std = NN_params[9]
    return BNNET


def adam_validate_variances(tbo_pars):
    # For z_v
    indices = np.arange(0, tbo_pars[2].shape[0])
    problematic_zs = indices[
        ((np.abs(tbo_pars[2] / tbo_pars[3]) > self.TH_Z) == True)[:, 0]]
    if problematic_zs.shape[0] != 0:
        tbo_pars[3][problematic_zs] = self.CAP_Z
        print('Lower-bounded', problematic_zs.shape[0],
              'random feature variances to', self.CAP_Z, '.')

    # For w_v
    no = 0
    for mesh in tbo_pars[1]:
        no += 1
        for row in range(0, mesh.shape[0]):
            for col in range(0, mesh.shape[1]):
                if mesh[row, col] <= self.CAP_W:
                    mesh[row, col] = self.CAP_W
                    print('Lower-bounded a weight variance to', self.CAP_W,
                          'in mesh', no, 'at index', row, ',', col, '.')

    return tbo_pars


def adam_training_statistics(prior_pars, tbo_pars):
    delta = []
    gain = 100000
    for par_bef, par_now in zip(prior_pars, tbo_pars):
        if type(par_bef) == list:
            d = 0
            for mesh in range(0, len(par_bef)):
                d = d + np.mean(np.abs(par_bef[mesh] - par_now[mesh]))
            d = d / len(par_bef)
        else:
            d = np.mean(np.abs(par_bef - par_now))
        delta.append(np.round(d * gain, 2))
    print('AVG update x', gain, 'per parameterset: ', delta[0], delta[1],
          delta[2], delta[3], delta[4])


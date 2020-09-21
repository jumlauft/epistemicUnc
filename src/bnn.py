# https://matthewmcateer.me/blog/a-quick-intro-to-bayesian-neural-networks/
# https://github.com/tensorflow/probability/issues/815
import numpy as np
from epimodel import EpiModel
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, InputLayer, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras


class BNN(EpiModel):
    def __init__(self, N_HIDDEN=10, TRAIN_EPOCHS=25, LEARNING_RATE=0.01,
                 N_SAMPLES=100, **kwargs):
        super().__init__(**kwargs)

        self.N_SAMPLES = N_SAMPLES
        self.TRAIN_EPOCHS = TRAIN_EPOCHS
        soft_0 = .01
        klw = 1/200
        def prior_trainable(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1),
                    reinterpreted_batch_ndims=1)),
            ])

        def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            c = np.log(np.expm1(1.))
            return tf.keras.Sequential([
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :n],
                               scale=1e-5 + 0.02 * tf.nn.softplus(
                                   c + t[..., n:])),
                    reinterpreted_batch_ndims=1)),
            ])

        # %% Model definition
        self.model = tfk.models.Sequential([
            tfp.layers.DenseVariational(N_HIDDEN, activation="relu", input_shape=[self.DX],
                                        make_posterior_fn=posterior_mean_field,
                                        make_prior_fn=prior_trainable,
                                        kl_weight=klw,
                                        ),
            tfp.layers.DenseVariational(N_HIDDEN, activation="relu",
                                        make_posterior_fn=posterior_mean_field,
                                        make_prior_fn=prior_trainable,
                                        kl_weight=klw,
                                        ),
            tfp.layers.DenseVariational(self.DY + 1, activation="linear",
                                        make_posterior_fn=posterior_mean_field,
                                        make_prior_fn=prior_trainable,
                                        kl_weight=klw,
                                        ),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(loc=t[..., :1],
                                     scale=1e-5 + tf.math.softplus(
                                         soft_0 * t[..., 1:]))),
        ])
        self.model.compile(loss=lambda y, yhat: -yhat.log_prob(y),
                      optimizer=tfk.optimizers.Adam(LEARNING_RATE))
    def train(self, xtr, ytr, display_progress = False):
        history = self.model.fit(xtr, ytr, epochs=self.TRAIN_EPOCHS,
                                 verbose=int(display_progress))
        return history.history['loss']

    def predict(self, x):
        Nte = x.shape[0]
        Yte = self.model.predict(np.tile(x, (self.N_SAMPLES, 1))).reshape(
            self.N_SAMPLES, Nte, self.DY)
        Yte_std = Yte.std(axis=0)
        Yte_mean = Yte.mean(axis=0)
        return Yte_mean, MinMaxScaler().fit_transform(Yte_std)



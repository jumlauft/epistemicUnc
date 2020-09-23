# rebuild from https://github.com/yaringal/DropoutUncertaintyDemos/blob/master/convnetjs/regression_uncertainty.js
import numpy as np
from src.epimodel import EpiModel
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.preprocessing import MinMaxScaler
from tensorflow import optimizers as tfo


class PermaDropout(tf.keras.layers.Layer):
    """Always-on dropout layer, i.e. it does not respect the training flag set
    to true by model.fit and false by model.predict. Unlike
    tf.keras.layers.Dropout, this layer does not return input unchanged if
    training=false, but alwaysrandomly drops a fraction self.rate of the input
    nodes.
    """

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs):
        return tf.nn.dropout(inputs, self.rate)

    def get_config(self):
        """enables model.save and restoration through
        tf.keras.models.load_model"""
        config = super().get_config()
        config["rate"] = self.rate
        return config


class Dropout(EpiModel):
    def __init__(self, N_HIDDEN=10, TRAIN_EPOCHS=10, LEARNING_RATE=0.01,
                 N_SAMPLES=10, DROPOUT_RATE=0.05, **kwargs):
        """

        Args:
            N_HIDDEN: Number of nodes per hidden layer(2)
            TRAIN_EPOCHS: number of training epochs
            LEARNING_RATE: learning rate for RMSprop algorithm
            N_SAMPLES: number of samples used for prediction
            DROPOUT_RATE: probability for turning a node off
            **kwargs:
        """
        super().__init__(**kwargs)

        self.N_SAMPLES = N_SAMPLES
        self.TRAIN_EPOCHS = TRAIN_EPOCHS

        tf.random.set_seed(0)
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(self.DX,)))
        self.model.add(PermaDropout(rate=DROPOUT_RATE))
        self.model.add(Dense(N_HIDDEN, activation="relu"))
        self.model.add(PermaDropout(rate=DROPOUT_RATE))
        self.model.add(Dense(N_HIDDEN, activation="relu"))
        self.model.add(Dense(self.DY))
        self.model.compile(optimizer=tfo.RMSprop(learning_rate=LEARNING_RATE),
                           loss='mean_squared_error')

    def train(self, xtr, ytr, display_progress=False):
        """ Training the model

        Args:
            xtr: input training points
            ytr: output training points
            display_progress: boolean

        Returns:
            list: loss over epochs
        """
        history = self.model.fit(xtr, ytr, epochs=self.TRAIN_EPOCHS,
                                 verbose=int(display_progress))
        return history.history['loss']

    def predict(self, x):
        """ predicts model output and epistemic uncertainty

        Args:
            x: inputs

        Returns:
            mean: prediction
            epi: epistemic uncertainty estimate
        """
        nte = x.shape[0]
        xt = np.tile(x, (self.N_SAMPLES, 1))
        yte = self.model.predict(xt).reshape(self.N_SAMPLES, nte, self.DY)
        yte_std = yte.std(axis=0)
        yte_mean = yte.mean(axis=0)
        return yte_mean, MinMaxScaler().fit_transform(yte_std)

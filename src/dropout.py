#rebuild from https://github.com/yaringal/DropoutUncertaintyDemos/blob/master/convnetjs/regression_uncertainty.js
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, InputLayer, Activation
from utils import scale_to_unit, weighted_RMSE

class PermaDropout(tf.keras.layers.Layer):
    """Always-on dropout layer, i.e. it does not respect the training flag set to
    true by model.fit and false by model.predict. Unlike tf.keras.layers.Dropout,
    this layer does not return input unchanged if training=false, but always
    randomly drops a fraction self.rate of the input nodes.
    """

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs):
        return tf.nn.dropout(inputs, self.rate)

    def get_config(self):
        """enables model.save and restoration through tf.keras.models.load_model"""
        config = super().get_config()
        config["rate"] = self.rate
        return config

class Dropout:
    N_HIDDEN = 50
    TRAIN_EPOCHS = 50
    LEARNING_RATE = 0.01
    MOMENTUM = 0.0001
    N_SAMPLES = 1000
    DROPOUT_RATE = 0.05
    BATCH_SIZE = 10
    def __init__(self, dx, dy):
        self.DX = dx
        self.DY = dy
        self.model = self._setup_nn()
    def _setup_nn(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.DX,)))
        model.add(PermaDropout(rate = self.DROPOUT_RATE))
        model.add(Dense(self.N_HIDDEN,activation="relu"))
        model.add(PermaDropout(rate = self.DROPOUT_RATE))
        model.add(Dense(self.N_HIDDEN,activation="relu"))
        model.add(Dense(self.DY))

        model.compile(optimizer=tf.optimizers.RMSprop(learning_rate = self.LEARNING_RATE), #SGD ,momentum = 0.0, decay=1e-5
              loss='mean_squared_error')
        return model
    def train(self):

        history = self.model.fit(self.Xtr, self.Ytr, epochs=self.TRAIN_EPOCHS,
                                 batch_size = self.BATCH_SIZE, verbose=0)
        return history.history['loss']

    def predict(self,x):
        def sigmoid(x):
            return 2 / (1 + np.exp(-0.1*x)) -1
        Nte = x.shape[0]
        Yte = self.model.predict(np.tile(x,(self.N_SAMPLES,1))).reshape(self.N_SAMPLES, Nte, self.DY)
        Yte_std = Yte.std(axis = 0)
        Yte_mean = Yte.mean(axis = 0)
        return Yte_mean, scale_to_unit(Yte_std)

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

    def weighted_RMSE(self,xte,yte):
        ypred, epi = self.predict(xte)
        return weighted_RMSE(yte,ypred, epi)
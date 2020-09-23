#rebuild from https://github.com/yaringal/DropoutUncertaintyDemos/blob/master/convnetjs/regression_uncertainty.js
import numpy as np
from src.epimodel import EpiModel
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, InputLayer, Activation
from sklearn.preprocessing import MinMaxScaler


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

class Dropout(EpiModel):
    def __init__(self, N_HIDDEN = 10, TRAIN_EPOCHS = 10, LEARNING_RATE = 0.01,
                 N_SAMPLES = 10, DROPOUT_RATE = 0.05,  **kwargs):
        super().__init__(**kwargs)
       
        self.N_SAMPLES = N_SAMPLES
        self.TRAIN_EPOCHS = TRAIN_EPOCHS
        
        tf.random.set_seed(0)
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(self.DX,)))
        self.model.add(PermaDropout(rate = DROPOUT_RATE))
        self.model.add(Dense(N_HIDDEN,activation="relu"))
        self.model.add(PermaDropout(rate = DROPOUT_RATE))
        self.model.add(Dense(N_HIDDEN,activation="relu"))
        self.model.add(Dense(self.DY))
        self.model.compile(optimizer=tf.optimizers.RMSprop(learning_rate = LEARNING_RATE), 
                           loss='mean_squared_error')

    def train(self, xtr, ytr, display_progress = False):
        history = self.model.fit(xtr, ytr, epochs=self.TRAIN_EPOCHS,
                                 verbose=int(display_progress))
        return history.history['loss']

    def predict(self,x):
        Nte = x.shape[0]
        Yte = self.model.predict(np.tile(x,(self.N_SAMPLES,1))).reshape(self.N_SAMPLES, Nte, self.DY)
        Yte_std = Yte.std(axis = 0)
        Yte_mean = Yte.mean(axis = 0)
        return Yte_mean, MinMaxScaler().fit_transform(Yte_std)



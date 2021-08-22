from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


"""This module is implmenting Time2Vec from https://arxiv.org/abs/1907.05321.
"""

class Time2Vec(Layer):
    def __init__(self, kernel_size, periodic_activation='sin'):
        super(Time2Vec, self).__init__(
            trainable=True,
            name='Time2VecLayer_'+periodic_activation.upper()
        )
        
        self.k = kernel_size
        self.p_activation = periodic_activation
    
    def build(self, input_shape):
        self.wb = self.add_weight(
            shape=(1, 1),
            initializer='uniform',
            trainable=True
        )
        
        self.bb = self.add_weight(
            shape=(1, 1),
            initializer='uniform',
            trainable=True
        )
        
        # Else needs to pass the periodic activation
        self.wa = self.add_weight(
            shape=(1, self.k),
            initializer='uniform',
            trainable=True
        )
        
        self.ba = self.add_weight(
            shape=(1, self.k),
            initializer='uniform',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        bias = self.wb * inputs + self.bb
        wgts = K.sin(K.dot(inputs, self.wa) + self.ba)
        return K.concatenate([bias, wgts], -1)
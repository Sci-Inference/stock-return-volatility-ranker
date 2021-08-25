import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model, Progbar
from tensorflow.keras import layers, Model, Input



class RankNet_Head(Model):
    def __init__(self,process_activation='linear',output_activation = 'sigmoid'):
        super().__init__()
        self.o = layers.Dense(1, activation=process_activation)
        self.oi_minus_oj = layers.Subtract()
        self.activation = output_activation
    
    def call(self, inputs):
        xi, xj = inputs
        oi = self.o(xi)
        oj= self.o(xj)
        oij = self.oi_minus_oj([oi, oj])
        output = layers.Activation(self.activation)(oij)
        return output




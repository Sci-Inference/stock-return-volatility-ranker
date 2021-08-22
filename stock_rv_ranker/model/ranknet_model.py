import tensorflow as tf
import numpy as np





class RankNet_Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, data):
        return super().train_step(data)
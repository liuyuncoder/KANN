import tensorflow as tf
import numpy as np
from InteracterLayer import InteracterLayer
import logging
import os
logging.basicConfig(level="ERROR")


class Interacter(tf.keras.layers.Layer):

    # - num_layers: the number of InteracterLayers.
    def __init__(self, num_layers, d_model, num_heads, dff, rate):
        super(Interacter, self).__init__()
        self.d_model = d_model

        self.enc_layers = [InteracterLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]


    def call(self, q, k, v, training, mask):
        num = q.shape.ndims

        attention_weights = {}  # for saving the attention weights of Interacter layer.
        # encoding based on InteracterLayer.
        for i, enc_layer in enumerate(self.enc_layers):
            x, attn = enc_layer(q, k, v, training, mask)
            # saving the attention weights obtained from Interacter layer.
            attention_weights['attention{}'.format(i + 1)] = attn
        return x, attention_weights

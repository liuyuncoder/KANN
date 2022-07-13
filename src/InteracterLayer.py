import numpy as np
import tensorflow as tf
from MultiHeadAttention import MultiHeadAttention
from tensorflow.keras import regularizers
import logging
logging.basicConfig(level="ERROR")


def point_wise_feed_forward_network(d_model, dff):
    # FFN do two linear projection，and one ReLU activation func.
    # kernel_regularizer=regularizers.l2(0.01)
    return tf.keras.Sequential([
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# Interacter has N InteracterLayers，and each InteracterLayer has two sub-layers: MHA & FFN
class InteracterLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(InteracterLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # layer norm
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, q, k, v, training, mask):
        # shape: (batch_size, input_seq_len, d_model)
        # attn.shape == (batch_size, num_heads, input_seq_len, input_seq_len)

        # sub-layer 1: MHA
        # padding mask to mask pad token
        attn_output, attn = self.mha(q, k, v, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(q + attn_output)

        # sub-layer 2: FFN
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(
            ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attn

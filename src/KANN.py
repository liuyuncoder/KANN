import tensorflow as tf
from Interacter import Interacter
import os
import numpy as np
from tensorflow.keras import regularizers
import logging
logging.basicConfig(level="ERROR")


# Using tf.keras.Model to build a model.
class KA(tf.keras.Model):
    def __init__(self, user_num, item_num, num_layers, d_model, num_heads, dff, rate=0):
        super(KA, self).__init__()

        self.d_model = d_model
        self.user_bias = tf.Variable([ 0.1 for _ in range(user_num + 2) ])
        self.item_bias = tf.Variable([ 0.1 for _ in range(item_num + 2) ])
        self.global_bias = tf.Variable(0.1)
        self.layernorm_conq = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_conk = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.attention_u = Interacter(
            num_layers, 4*d_model, num_heads, dff, rate)

        self.attention_i = Interacter(
            num_layers, 4*d_model, num_heads, dff, rate)

        self.interacter_u = Interacter(
            num_layers, 4*d_model, num_heads, dff, rate)

        self.interacter_i = Interacter(
            num_layers, 4*d_model, num_heads, dff, rate)
        self.layernorm_con_oriu = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_con_orii = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.after_inter_layer = tf.keras.layers.Dense(d_model, activation='relu')

        self.prediction_u_layer = tf.keras.layers.Dense(1, activation='relu')
        self.prediction_i_layer = tf.keras.layers.Dense(1, activation='relu')
        self.prediction_layer = tf.keras.layers.Dense(1, activation='relu')

    def call(self, u_batch, i_batch, training, uid, iid, enc_padding_mask, dec_padding_mask):

        data_dir = "../data/imdb"
        tri50_dir = os.path.join(data_dir, 'entity_emb50_concat_mean.bin')
        re_vec50 = np.memmap(tri50_dir, dtype='float32', mode='r')
        column_num50 = self.d_model*3
        re_row_num50 = int(re_vec50.shape[0]/column_num50)
        re_vec50 = np.resize(re_vec50, (re_row_num50, column_num50))

        original_e50_dir = os.path.join(data_dir, 'entity_oriemb50.bin')
        ori_e50 = np.memmap(original_e50_dir, dtype='float32', mode='r')
        column_num50_ori = self.d_model
        ori_e_row50 = int(ori_e50.shape[0]/column_num50_ori)
        ori_e50 = np.resize(ori_e50, (ori_e_row50, column_num50_ori))

        q_context = tf.nn.embedding_lookup(re_vec50, u_batch)
        k_context = tf.nn.embedding_lookup(re_vec50, i_batch)

        q_ori = tf.nn.embedding_lookup(ori_e50, u_batch) # (batch_size, seq_len, emb_size)
        k_ori = tf.nn.embedding_lookup(ori_e50, i_batch)

        ureview_emb = self.layernorm_conq(tf.concat([q_context, q_ori], 2))
        ireview_emb = self.layernorm_conk(tf.concat([k_context, k_ori], 2))

        ureview_emb *= tf.math.sqrt(tf.cast(self.d_model*4, tf.float32))
        ireview_emb *= tf.math.sqrt(tf.cast(self.d_model*4, tf.float32))

        ureview_emb = self.dropout(ureview_emb, training=training)
        ireview_emb = self.dropout(ireview_emb, training=training)

        attention_weights = {}  # used to save the attention weighnts of inner and outer attention mechanisms

        u_out, attn_u = self.attention_u(
            ureview_emb, ureview_emb, ureview_emb, training, enc_padding_mask)
        i_out, attn_i = self.attention_i(
            ireview_emb, ireview_emb, ireview_emb, training, dec_padding_mask)

        inter_out_u, attn_ui = self.interacter_u(
            u_out, i_out, i_out, training, dec_padding_mask)
        inter_out_i, attn_iu = self.interacter_i(
            i_out, u_out, u_out, training, enc_padding_mask)

        attention_weights['attn_u'] = attn_u
        attention_weights['attn_i'] = attn_i
        attention_weights['attn_ui'] = attn_ui
        attention_weights['attn_iu'] = attn_iu

        # concat original features
        inter_out_u = self.layernorm_con_oriu(tf.concat([inter_out_u, q_ori], 2))
        inter_out_i = self.layernorm_con_orii(tf.concat([inter_out_i, k_ori], 2))
        inter_out_u = tf.transpose(inter_out_u, perm=[0, 2, 1])
        inter_out_i = tf.transpose(inter_out_i, perm=[0, 2, 1])
        out_u = self.prediction_u_layer(
            inter_out_u)  # (batch_size, 5*d_model, 1)
        out_u = tf.reshape(out_u, [-1, self.d_model*5])
        out_i = self.prediction_i_layer(inter_out_i)
        out_i = tf.reshape(out_i, [-1, self.d_model*5])

        interaction = tf.multiply(out_u, out_i)  # (batch_size, d_model)
        interaction = self.after_inter_layer(interaction)

        # Output the interaction result through a linear layer               could be an nonlinear。。。。。。。。。。
        scores = self.prediction_layer(interaction)  # (batch_size, 1)
        user_bias = tf.nn.embedding_lookup(self.user_bias, uid)
        item_bias = tf.nn.embedding_lookup(self.item_bias, iid)
        scores = (scores + user_bias + item_bias + self.global_bias)

        return scores, attention_weights

"""
https://rubikscode.net/2019/08/05/transformer-with-python-and-tensorflow-2-0-attention-layers/
"""

import tensorflow as tf 
from tensorflow.layers import Dense
import os

training = bool(os.environ["TRAINING"])

class ScaledDotProduct:

    def calculate_output_weights(self, q, k, v, mask):    
        qk = tf.matmul(q, k, transpose_b=True)    
        dk = tf.cast(tf.shape(k)[–1], tf.float32)    
        scaled_attention = qk / tf.math.sqrt(dk)        
        if mask is not None:    
            scaled_attention_logits += (mask * –1e9)         
            weights = tf.nn.softmax(scaled_attention, axis=–1)    
            output = tf.matmul(weights, v)               
        return output, weights


class MultiHeadedAttention(tf.keras.layers.layer):

    __slots__ = ['input_dim', 'num_heads', 'attention_layer', 'depth', 
                 'q_layer', 'k_layer', 'v_layer', 'linear_layer']

    def __init__(self, input_dim, num_heads, attention_layer):
        super(MultiHeadedAttention, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.attention_layer = attention_layer
        self.depth = self.input_dim // self.num_heads
        
        self.q_layer = Dense(input_dim)
        self.k_layer = Dense(input_dim)
        self.v_layer = Dense(input_dim)
        
        self.linear_layer = Dense(input_dim)

    def split(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # Run through linear layers
        q = self.q_layer(q)
        k = self.k_layer(k)
        v = self.v_layer(v)

        # Split the heads 
        q = self.split(q, batch_size)
        k = self.split(k, batch_size)
        v = self.split(v, batch_size)

        # Run through attention 
        attention_output, weights = self.attention_layer.calculate_output_weights(q, k, v, mask)

        # Prepare for the rest of processing
        output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.num_neurons))

        # Run through final linear layer
        output = self.linear_layer(concat_attention)

        return output, weights



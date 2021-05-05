import tensorflow as tf
import os
from MultiHeadAttention import MultiHeadAttention
# from tensorflow.keras.layers import MultiHeadAttention

batch_size = int(os.environ["BATCH_SIZE"])
training = bool(os.environ["TRAINING"])


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 rate=0.1):
        super(EncoderLayer, self).__init__()

        self.multi_headed_attention = MultiHeadAttention(d_model, num_heads)

        self.feed_forward_network = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask):
        # (1) - Attention Score
        attn_output, attn_weights = self.multi_headed_attention(x, x, x, mask)  # (batch_size, input_seq_len, d_model)

        # (2) - Add & Normalize
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        # (3) - Feed Forward NN
        feed_forward_output = self.feed_forward_network(out1)  # (batch_size, input_seq_len, d_model)

        # (4) - Add & Normalize
        feed_forward_output = self.dropout2(feed_forward_output, training=training)
        out2 = self.layernorm2(out1 + feed_forward_output)  # (batch_size, input_seq_len, d_model)

        return tf.convert_to_tensor(out2), tf.convert_to_tensor(attn_weights)


class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers,
                 d_model,
                 embedding_matrix,
                 num_heads,
                 dff,
                 input_vocab_size,
                 max_seq_len,
                 rate=0.1):
        super(TransformerBlock, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

    def call(self, x, mask):
        attn_weights = None
        for i in range(self.num_layers):
            x, attn_weights = self.enc_layers[i](x, mask)

        return tf.convert_to_tensor(x), tf.convert_to_tensor(attn_weights)  # (batch_size, input_seq_len, d_model)

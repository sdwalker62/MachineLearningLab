import tensorflow as tf
import os
from MultiHeadAttention import MultiHeadAttention
from PositionalEncoder import PositionalEncoding

batch_size = int(os.environ["BATCH_SIZE"])
training = bool(int(os.environ["TRAINING"]))

class EncoderLayer(tf.keras.layers.Layer):
  
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.multi_headed_attention = MultiHeadAttention(d_model, num_heads)

    self.feed_forward_network = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dropout(rate),
      tf.keras.layers.Dense(d_model), # (batch_size, seq_len, d_model)
      tf.keras.layers.Dropout(rate)
    ])

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)

  def call(self, x, mask):
    # (1) - Attention Score
    attn_output, attn_weights = self.multi_headed_attention(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    
    # (2) - Add & Normalize
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    # (3) - Feed Forward NN
    feed_forward_output = self.feed_forward_network(out1)  # (batch_size, input_seq_len, d_model)

    # (4) - Add & Normalize
    out2 = self.layernorm2(out1 + feed_forward_output)  # (batch_size, input_seq_len, d_model)

    return out2, attn_weights

class Encoder(tf.keras.layers.Layer):

  def __init__(self, 
               num_layers, 
               d_model, 
               embedding_matrix,
               num_heads, 
               dff, 
               input_vocab_size,
               max_seq_len,
               rate=0.1):
    
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.embedding = tf.keras.layers.Embedding(input_vocab_size,
                                                d_model,
                                                weights=[embedding_matrix],
                                                input_length=max_seq_len,
                                                trainable=True)

    self.pos_encoding = PositionalEncoding(batch_size, d_model)
    #self.pos_encoding = positional_encoding(maximum_position_encoding,
    #                                        self.d_model)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, mask):
    # input_seq_len == max_seq_len
    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    #x += self.pos_encoding[:, :seq_len, :]
    x = self.pos_encoding(x)
    x = self.dropout(x, training=training)

    attn_weights = None
    for i in range(self.num_layers):
      x, attn_weights = self.enc_layers[i](x, mask)

    return x, attn_weights  # (batch_size, input_seq_len, d_model)
import tensorflow as tf 
from TransformerBlock import TransformerBlock
from PositionalEncoder import PositionalEncoding
import os

training = bool(os.environ["TRAINING"])

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, embedding_matrix, max_seq_len, rate=0.1):
    super(Transformer, self).__init__()

    self.d_model = d_model

    # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.embedding = tf.keras.layers.Embedding(input_vocab_size,
                                                d_model,
                                                weights=[embedding_matrix],
                                                input_length=max_seq_len,
                                                trainable=True)

    self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
    #self.pos_encoding = positional_encoding(maximum_position_encoding,
    #                                        self.d_model)

    self.transformer_block = TransformerBlock(num_layers, d_model, embedding_matrix, num_heads, dff,
                             input_vocab_size, max_seq_len, rate)

    # self.decoder = Decoder(num_layers, d_model, num_heads, dff,
    #                        target_vocab_size, pe_target, rate)

    # self.transformer = tf.keras.Sequential([
    #   self.transformer_block,
    #   self.transformer_block,
    #   tf.keras.layers.Dense(target_vocab_size, activation='relu'),
    #   tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first'),
    #   tf.keras.layers.Softmax()
    # ])

    # self.final_layer = tf.keras.Sequential([
    #   tf.keras.layers.Dense(target_vocab_size),
    #   tf.keras.layers.Softmax()
    # ])

    self.pooling_layer = tf.keras.Sequential([
      tf.keras.layers.Dense(target_vocab_size, activation='relu'),
      tf.keras.layers.AveragePooling1D(data_format='channels_last'),
      tf.keras.layers.Softmax()
    ])

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, inp, tar, enc_padding_mask,
           look_ahead_mask, dec_padding_mask):

    x = inp
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    #x += self.pos_encoding[:, :seq_len, :]
    x = self.pos_encoding(x)
    x = self.dropout(x, training=training)

    # Transformer Block #1
    enc_output, _ = self.transformer_block(x, enc_padding_mask)  # (batch_size, inp_seq_len, d_model), (batch_size, class, inp_seq_len, inp_seq_len)

    print(f"Shape Block 1: {enc_output.shape}")


    # Transformer Block #2 vv (takes the place of the Decoder)
    fin_output, _ = self.transformer_block(enc_output, enc_padding_mask)

    print(f"Shape Block 2: {fin_output.shape}")
    # # dec_output.shape == (batch_size, tar_seq_len, d_model)
    # # dec_output, attention_weights = self.decoder(
    # #     tar, enc_output, training, look_ahead_mask, dec_padding_mask)

    # print(attention_weights.shape)
    # final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)
    final_output = self.pooling_layer(fin_output) # (batch_size, max_seq_len, class)

    print(f"Shape Block 3: {final_output.shape}")
    return final_output#, attention_weights
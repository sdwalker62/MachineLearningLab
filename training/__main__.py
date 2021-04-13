from Transformer import Transformer
from MultiHeadAttention import MultiHeadAttention

from Metrics import loss_function
from Metrics import accuracy_function

import tensorflow as tf
import os
import numpy as np

layers = int(os.environ["TRANSFORMER_LAYERS"])
d_model = int(os.environ["W2V_EMBED_SIZE"])
dff = int(os.environ["TRANSFORMER_DFF"])
heads = int(os.environ["TRANSFORMER_HEADS"])
batch_size = int(os.environ["BATCH_SIZE"])
training = bool(os.environ["TRAINING"])

optimus_prime = None



train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

  with tf.GradientTape() as tape:
    predictions = optimus_prime(inp, tar_inp,
                                  None, None, None)

    print(predictions)
    # loss = loss_function(tar_real, predictions)

  # gradients = tape.gradient(loss, transformer.trainable_variables)
  # optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  # train_loss(loss)
  # train_accuracy(accuracy_function(tar_real, predictions))

if __name__ == '__main__':

    # temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    # y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    # out, attn = temp_mha(y, k=y, q=y, mask=None)
    # out.shape, attn.shape

    optimus_prime = Transformer(layers, d_model, heads, dff, 1800, 2, 10000, 6000)

    temp_input = tf.random.uniform((64, 38))
    temp_target = tf.random.uniform((64, 36))

    train_step(temp_input, temp_target)
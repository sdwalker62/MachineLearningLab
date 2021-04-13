from Transformer import Transformer
from MultiHeadAttention import MultiHeadAttention

from Metrics import loss_function
from Metrics import accuracy_function

import tensorflow as tf
import os
import numpy as np
import joblib

layers = int(os.environ["TRANSFORMER_LAYERS"])
d_model = int(os.environ["W2V_EMBED_SIZE"])
dff = int(os.environ["TRANSFORMER_DFF"])
heads = int(os.environ["TRANSFORMER_HEADS"])
batch_size = int(os.environ["BATCH_SIZE"])
training = bool(os.environ["TRAINING"])

optimus_prime = None

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)
]

# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  # tar_inp = tar[:, :-1]
  tar_inp = None
  # tar_real = tar[:, 1:]

  # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  attn_weights = []
  with tf.GradientTape() as tape:
    predictions, attn_scores = optimus_prime(inp, tar_inp,
                                  None, None, None)

    attn_weights.append(attn_scores)
    print(predictions)
    # loss = loss_function(tar_real, predictions)
  return attn_weights

  # gradients = tape.gradient(loss, transformer.trainable_variables)
  # optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  # train_loss(loss)
  # train_accuracy(accuracy_function(tar_real, predictions))

if __name__ == '__main__':

    word_embeddings = joblib.load("/results/w2v_weights.joblib")
    vocabulary = joblib.load("/results/vocab_dict.joblib")

    dataset = tf.data.experimental.SqlDataset("sqlite", "/database/elastic_logs.db", "SELECT * FROM logs", (tf.string, tf.string, tf.string, tf.string))
    logs = []

    for element in dataset.as_numpy_iterator():
      sequence = []
      for word in element[2].split():
        word = word.decode('UTF-8')
        if word in vocabulary.keys():
          sequence.append(vocabulary[word])
        else:
          sequence.append(0)

      logs.append(sequence)

    joblib.dump(logs, "/results/sequence_indices.joblib")

    vocab_size = len(vocabulary)

    optimus_prime = Transformer(layers, d_model, heads, dff, vocab_size, 4, word_embeddings, 50)

    learning_rate = CustomSchedule(d_model)

    custom_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    optimus_prime.compile(optimizer=custom_optimizer, loss=loss_function)

    # temp_input = tf.random.uniform((64, 38))
    # temp_target = tf.random.uniform((64, 4))

    attn = train_step(logs, None)
    # print(attn)
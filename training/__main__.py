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
    print(f'Sending in Matrix of size {inp.shape}')
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

def ProcessLogs(file_path, vocabulary, max_seq_len, override=False):
  if os.path.exists(file_path) and override == False:
    return joblib.load(file_path)
  else:
    dataset = tf.data.experimental.SqlDataset("sqlite", "/database/elastic_logs.db", "SELECT * FROM logs", (tf.string, tf.string, tf.string, tf.string))
    batch_size = len(list(dataset.as_numpy_iterator()))
    logs = np.zeros((batch_size, max_seq_len))

    for log_idx, element in enumerate(dataset.as_numpy_iterator()):
      for seq_idx, word in enumerate(element[2].split()):
        if seq_idx > max_seq_len:
          break
        word = word.decode('UTF-8')
        if word in vocabulary.keys():
          logs[log_idx, seq_idx] = vocabulary[word]
        else:
          sequence.append(0)

      if max_seq_len - len(sequence) > 0:
        for seq_idx in range(len(sequence), max_seq_len):
          logs[log_idx, seq_idx] = 0
          
    joblib.dump(logs, file_path)
    return logs


if __name__ == '__main__':

    word_embeddings = joblib.load("/results/w2v_weights.joblib")
    vocabulary = joblib.load("/results/vocab_dict.joblib")
    logs = ProcessLogs("/results/sequence_indices.joblib", vocabulary, True)

    vocab_size = len(vocabulary)

    optimus_prime = Transformer(layers, d_model, heads, dff, vocab_size, 4, word_embeddings, max_seq_len)

    learning_rate = CustomSchedule(d_model)

    custom_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    optimus_prime.compile(optimizer=custom_optimizer, loss=loss_function)

    # temp_input = tf.random.uniform((64, 38))
    # temp_target = tf.random.uniform((64, 4))

    logs = np.array(logs)
    # logs = tf.convert_to_tensor(logs)
    print(logs.shape)

    # attn = train_step(logs, None)
    # print(attn)
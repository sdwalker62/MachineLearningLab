import tensorflow as tf
import numpy as np
import os

loss_object = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True)

batch_size = int(os.environ["BATCH_SIZE"])

def grad(model, x):
    with tf.GradientTape() as tape:
      predictions = model(x)

      y_seq_pred = np.empty((batch_size, 4))
      y_true = x[1]

      for idx in range(batch_size):
          seq_pred = predictions[idx]  # (max_seq_len, classifications)
          seq_pred = np.array(seq_pred)
          y_seq_pred[idx] = seq_pred.mean(axis=0)

    loss = loss_function(y_true, y_seq_pred)
    grads = tape.gradient(loss, model.trainable_variables)

    return loss, grads

def loss_function(real, pred):
  return loss_object(real, pred)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

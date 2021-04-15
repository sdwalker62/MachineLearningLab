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
          # seq_pred = seq_pred.to_numpy_array()
          # y_seq_pred[idx] = tf.math.reduce_mean(seq_pred)
          # seq_pred = np.ar
          onebyfour = tf.math.reduce_mean(seq_pred, axis=0)
          print(onebyfour)
          # y_seq_pred[idx] = tf.reduce_mean(seq_pred, axis=0)

    loss = loss_function(y_true, predictions)
    grads = tape.gradient(loss, model.trainable_variables)

    return loss, grads, y_seq_pred


def loss_function(real, pred):
    return loss_object(real, pred)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

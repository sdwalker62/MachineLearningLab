import tensorflow as tf
import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import os

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)

batch_size = int(os.environ["BATCH_SIZE"])

lr = LogisticRegression()

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
    lr.fit(pred, real)
    loss = tf.cast(log_loss(real, lr.predict_proba(pred), eps=1e-15), dtype=tf.float32)
    return loss

def loss_function2(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)



def accuracy_function(real, pred):
    real = tf.cast(real, tf.int64)
    pred = tf.argmax(pred, axis=1)
    accuracies = tf.equal(real, pred)

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

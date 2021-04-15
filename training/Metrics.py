import tensorflow as tf

loss_object = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True)

def grad(model, real, pred):
    loss = loss_function(labels, y_seq_pred)
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

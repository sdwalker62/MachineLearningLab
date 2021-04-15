from Transformer import Transformer
from MultiHeadAttention import MultiHeadAttention
from tqdm import tqdm
from Metrics import grad
from Metrics import loss_function
from Metrics import accuracy_function
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelBinarizer
from einops import rearrange

import tensorflow as tf
import os
import numpy as np
import joblib
import pandas as pd
import logging
import sqlite3 as sql
import time

num_layers = int(os.environ["TRANSFORMER_LAYERS"])
d_model = int(os.environ["W2V_EMBED_SIZE"])
dff = int(os.environ["TRANSFORMER_DFF"])
num_heads = int(os.environ["TRANSFORMER_HEADS"])
batch_size = int(os.environ["BATCH_SIZE"])
training = bool(int(os.environ["TRAINING"]))
epochs = int(os.environ["EPOCHS"])
max_seq_len = 200


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model: int, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimus_prime = None
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
adm_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

epoch_loss = tf.keras.metrics.Mean(name='train_loss')
epoch_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

train_step_signature = [
    tf.TensorSpec(shape=(batch_size, max_seq_len), dtype=tf.int64),
    tf.TensorSpec(shape=(batch_size, 4), dtype=tf.int64)
]

@tf.function(input_signature=train_step_signature)
def train_step(log_batch: tf.Tensor, labels: tf.Tensor):

    transformer_input = tf.tuple([
        log_batch,  # <tf.Tensor: shape=(batch_size, max_seq_len), dtype=uint32>
        labels  # <tf.Tensor: shape=(batch_size, num_classes), dtype=uint32>
    ])
    
    # print("CALLING OP")
    predictions = optimus_prime(transformer_input)

        # print((predictions))


        # print(type(predictions[0]))

    return predictions



logging.basicConfig(format='%(asctime)s %(levelname)s | %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def database_builder(path: str) -> pd.DataFrame():
    logger.info('Building DataFrame ...')
    (_, _, files) = next(os.walk(path))
    sql_query = 'SELECT * FROM logs'
    data = []
    for f in files:
        if '.db' in f:
            conn = create_connection(path + f)
            d = pd.read_sql_query(sql_query, conn)
            data.append(d)
    logger.info('...complete!')
    return pd.concat(data)


def create_connection(path: str) -> sql.Connection:
    """
    Creates a database connection
    :param path: str
        path to database object
    :return sql.Connection
        a connection to the database
    """
    try:
        conn = sql.connect(path)
        logger.info('Connected to database ' + path)
        return conn
    except sql.Error as e:
        logger.warning(e)


def get_max_length_(dataset: pd.DataFrame, buffer_size: float) -> int:
    return int((1 + buffer_size) * dataset['log'].str.len().max())


def process_batch(dataset: pd.DataFrame,
                  vocabulary: dict,
                  max_seq_len: int,
                  idx: int,
                  labels: dict) -> tuple:
    logs = np.zeros((batch_size, max_seq_len))
    y_true = np.empty((batch_size, 4))

    start_window = idx * batch_size
    end_window = (idx + 1) * batch_size
    for log_idx, log in enumerate(dataset['log'][start_window:end_window]):
        for seq_idx, word in enumerate(log.split()):
            logs[log_idx, seq_idx] = vocabulary[word] if word in vocabulary.keys() else 0
        y_true[log_idx] = labels[dataset['label'][log_idx]]

    return tf.convert_to_tensor(logs, dtype=tf.int64), tf.convert_to_tensor(y_true, dtype=tf.int64)


if __name__ == '__main__':

    logging.info('Loading assets')
    word_embedding_matrix = joblib.load("/results/w2v_weights.joblib")
    vocabulary = joblib.load("/results/vocab_dict.joblib")
    dataset = database_builder('/database/')
    max_seq_len = 200  # get_max_length_(dataset, 0.0)
    vocab_size = len(vocabulary)

    logging.info('Processing logs for training')
    label_unique = dataset['label'].unique()
    lbp = LabelBinarizer().fit(label_unique)
    binary_labels = lbp.transform(label_unique)

    labels = {}
    for idx, label in enumerate(label_unique):
        labels.update({
            label: binary_labels[idx]
        })

    n_logs = len(dataset.index)
    n_iter = 5  # n_logs // batch_size
    remainder = n_logs % batch_size
    attns = []

    optimus_prime = Transformer(
        num_layers,
        d_model,
        num_heads,
        dff,
        vocab_size,
        word_embedding_matrix,
        max_seq_len,
        rate=0.1)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    checkpoint_path = "./checkpoints/train"
    checkpoint = tf.train.Checkpoint(transformer=optimus_prime, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    for epoch in tqdm(range(epochs)):

        start = time.time()

        for idx in range(n_iter):
            log_batch, labels = process_batch(dataset, vocabulary, max_seq_len, idx, labels)
            
            with tf.GradientTape() as tape:
                
                # Returns Eager Tensor for Predictions
                print("RUNNING TRAING STEP")
                preds = train_step(log_batch, labels)

                y_seq_pred = np.empty((batch_size, 4))
                for idx in range(batch_size):
                    seq_pred = preds[idx]  # (max_seq_len, classifications)
                    y_seq_pred[idx] = tf.reduce_mean(seq_pred, axis=0)

                loss = loss_function(labels, y_seq_pred)
                grads = tape.gradient(loss, optimus_prime.trainable_variables)

            # Optimize the model
            adm_optimizer.apply_gradients(zip(grads, optimus_prime.trainable_variables))

            # Tracking Progress
            epoch_loss.update_state(loss)  # Adding Batch Loss
            epoch_accuracy.update_state(labels, y_seq_pred)

            print(f'Epoch {epoch + 1} Batch {idx + 1}')
            # print(f'Epoch {epoch + 1} Batch {idx} Loss {epoch_loss.result():.4f} Accuracy {epoch_accuracy.result():.4f}')

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss.result(),
                                                                epoch_accuracy.result()))

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
    # joblib.dump(attns, "/results/5_of_50_attn_weights.joblib")

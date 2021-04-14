from Transformer import Transformer
from MultiHeadAttention import MultiHeadAttention
from tqdm import tqdm
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

layers = int(os.environ["TRANSFORMER_LAYERS"])
d_model = int(os.environ["W2V_EMBED_SIZE"])
dff = int(os.environ["TRANSFORMER_DFF"])
heads = int(os.environ["TRANSFORMER_HEADS"])
batch_size = int(os.environ["BATCH_SIZE"])
training = bool(int(os.environ["TRAINING"]))
epochs = int(os.environ["EPOCHS"])

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


learning_rate = CustomSchedule(d_model)

optimus_prime = None
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
adm_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# train_step_signature = [
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64)
# ]

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')


# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    # tar_inp = tar[:, :-1]
    tar_inp = None
    # tar_real = tar[:, 1:]

    # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    attn_weights = []
    with tf.GradientTape() as tape:
        # print(f'Sending in Matrix of size {inp.shape}')
        # attn_scores
        predictions = optimus_prime(inp, tar_inp,
                                    None, None, None)

        y_pred = predictions
        y_seq_pred = np.empty((batch_size, 4))
        y_true = tar
        cce = tf.keras.losses.CategoricalCrossentropy()

        for idx in range(batch_size):
            seq_pred = y_pred[idx] # (max_seq_len, classifications)
            seq_pred = np.array(seq_pred)
            y_seq_pred[idx] = seq_pred.mean(axis=0)

        loss = cce(y_true, y_seq_pred)

    # sgd_gradients = sgd_tape.gradient(loss, class_nn.trainable_variables)
    # sgd_optimizer.apply_gradients(zip(sgd_gradients, class_nn.trainable_variables))

    optimus_gradients = tape.gradient(loss, optimus_prime.trainable_variables)
    adm_optimizer.apply_gradients(zip(optimus_gradients, optimus_prime.trainable_variables))

    train_loss(loss)
    train_accuracy.update_state(y_true, y_seq_pred)
    return attn_weights


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
            conn = create_connection(path + '/' + f)
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
                  labels: dict,
                  override=False) -> tuple:

    logs = np.zeros((batch_size, max_seq_len))
    y_true = np.empty((batch_size,4))

    start_window = idx * batch_size
    end_window = (idx + 1) * batch_size
    for log_idx, log in enumerate(dataset['log'][start_window:end_window]):
        for seq_idx, word in enumerate(log.split()):
            logs[log_idx, seq_idx] = vocabulary[word] if word in vocabulary.keys() else 0
        y_true[log_idx] = labels[dataset['label'][log_idx]]

    return logs, y_true


if __name__ == '__main__':
    logging.info('Loading assets')
    word_embeddings = joblib.load("/results/w2v_weights.joblib")
    vocabulary = joblib.load("/results/vocab_dict.joblib")
    dataset = database_builder('/database/')
    max_seq_len = 512#get_max_length_(dataset, 0.0)
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
    n_iter = 5 #n_logs // batch_size
    remainder = n_logs % batch_size
    attns = []
    for epoch in tqdm(range(epochs)):
        
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for idx in range(n_iter):
            batch, y_true = process_batch(dataset, vocabulary, max_seq_len, idx, labels, True)

            optimus_prime = Transformer(layers, d_model, heads, dff, vocab_size, 4, word_embeddings, max_seq_len)

            optimus_prime.compile(optimizer=adm_optimizer, loss=tf.keras.losses.CategoricalCrossentropy)

            attn = train_step(batch, y_true)
            # attns.append(attn)

            print(f'Epoch {epoch + 1} Batch {idx} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
    # joblib.dump(attns, "/results/5_of_50_attn_weights.joblib")

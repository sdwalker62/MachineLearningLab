from Transformer import Transformer
from MultiHeadAttention import MultiHeadAttention
from tqdm import tqdm
from Metrics import loss_function
from Metrics import accuracy_function


import tensorflow as tf
import os
import numpy as np
import joblib
import pandas as pd
import logging
import sqlite3 as sql

layers = int(os.environ["TRANSFORMER_LAYERS"])
d_model = int(os.environ["W2V_EMBED_SIZE"])
dff = int(os.environ["TRANSFORMER_DFF"])
heads = int(os.environ["TRANSFORMER_HEADS"])
batch_size = int(os.environ["BATCH_SIZE"])
training = bool(int(os.environ["TRAINING"]))

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


def process_batch(file_path: str,
                  dataset: pd.DataFrame,
                  vocabulary: dict,
                  max_seq_len: int,
                  idx: int,
                  override=False) -> np.array:

    if os.path.exists(file_path) and not override:
        return joblib.load(file_path)
    else:
        logs = np.zeros((batch_size, max_seq_len))
        start_window = idx * batch_size
        end_window = (idx + 1) * batch_size
        for log_idx, log in enumerate(dataset['log'][start_window:end_window]):
            for seq_idx, word in enumerate(log.split()):
                logs[log_idx, seq_idx] = vocabulary[word] if word in vocabulary.keys() else 0

        joblib.dump(logs, file_path)
        return logs


if __name__ == '__main__':
    logging.info('Loading assets')
    word_embeddings = joblib.load("/results/w2v_weights.joblib")
    vocabulary = joblib.load("/results/vocab_dict.joblib")
    dataset = database_builder('/database/')
    max_seq_len = get_max_length_(dataset, 0.0)
    vocab_size = len(vocabulary)

    logging.info('Processing logs for training')

    n_logs = len(dataset.index)
    n_iter = n_logs // batch_size
    remainder = n_logs % batch_size

    for idx in tqdm(range(n_iter)):
        batch = process_batch("/results/sequence_indices.joblib", dataset, vocabulary, max_seq_len, idx, True)

        logging.info('Creating transformer and training')
        optimus_prime = Transformer(layers, d_model, heads, dff, vocab_size, 4, word_embeddings, max_seq_len)

        learning_rate = CustomSchedule(d_model)

        custom_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        optimus_prime.compile(optimizer=custom_optimizer, loss=loss_function)

        attn = train_step(batch, None)
        print(attn)

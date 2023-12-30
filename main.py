import time
import logging

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text

from transformer.__init__ import *

# === DOESN'T WORK YET ===
def download_dataset():
    model_name = 'ted_hrlr_translate_pt_en_converter'
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True
    )

    tokenizers = tf.saved_model.load(model_name) # requires tenorflow_text build
# ===  ===

# === FOR TRAINING ===
MAX_TOKENS=128
def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    return (pt, en_inputs), en_labels

BUFFER_SIZE = 20000
BATCH_SIZE = 64
def make_batches(ds):
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))


def main():
    # hyperparams
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    # aquire dataset
    tokenizers = download_dataset()

    # transformer setup
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        dropout_rate=dropout_rate
    )

    print(en.shape)                     # >>> (64, 58)
    print(pt.shape)                     # >>> (64, 62)
    print(transformer((pt, en)).shape)  # >>> (64, 58, 7010)

    # (batch, heads, target_seq, input_seq)
    print(transformer.decoder.decoder_layers[-1].cross_attention.last_scores.shape) # >>> (64, 8, 58, 62)

    # print model summary
    transformer.summary()

if __name__ == "__main__":
    main()

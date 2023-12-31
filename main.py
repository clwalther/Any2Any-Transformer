import time
import logging

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text

from transformer.__init__ import *
from datahandling.__init__ import *

# === TEMP ===
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            d_model,
            mask_zero=True
        )
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[tf.newaxis, :tf.shape(x)[1], :]
        return x

def positional_encoding(length, depth):
    depth /= 2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )

    return tf.cast(pos_encoding, dtype=tf.float32)
# ===  ===


def main():
    # hyperparams
    num_layers = 4
    d_model = 128
    num_heads = 8
    dff = 512
    dropout_rate = 0.1

    # aquire dataset
    tokenizers, en, pt = load_dataset()

    # setup encoder, decoder
    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        dropout_rate=dropout_rate
    )

    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        dropout_rate=dropout_rate
    )

    # config encoder, decoder
    encoder.set_pre_layer(
        pre_layer=PositionalEmbedding(
            vocab_size=tokenizers.pt.get_vocab_size().numpy(),
            d_model=d_model
        )
    )
    decoder.set_pre_layer(
        pre_layer=PositionalEmbedding(
            vocab_size=tokenizers.en.get_vocab_size().numpy(),
            d_model=d_model
        )
    )
    decoder.set_post_layer(
        post_layer=tf.keras.layers.Dense(
            tokenizers.en.get_vocab_size().numpy()
        )
    )

    # setup, config transformer
    transformer = Transformer()

    transformer.set_encoder(encoder=encoder)
    transformer.set_decoder(decoder=decoder)

    # === VERIFICATION ===
    print(en.shape)                     # >>> (64, 58)
    print(pt.shape)                     # >>> (64, 62)
    print(transformer((pt, en)).shape)  # >>> (64, 58, 7010)

    # (batch, heads, target_seq, input_seq)
    print(transformer.decoder.decoder_layers[-1].cross_attention.last_scores.shape) # >>> (64, 8, 58, 62)

    # print model summary
    transformer.summary()

if __name__ == "__main__":
    main()

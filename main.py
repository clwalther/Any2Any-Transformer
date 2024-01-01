import time
import logging

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text

from transformer.__init__ import *
from dataset.__init__ import *
from training.__init__ import *

# === PRE LAYERS (according to the transformer ressource) === #
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
# ===   === #


# === TRANSLATOR v1 === #
class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length):
        # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # As the output language is English, initialize the output with the
        # English `[START]` token.
        start_end = self.tokenizers.en.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            # Select the last token from the `seq_len` dimension.
            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i+1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # The output shape is `(1, tokens)`.
        text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.

        tokens = tokenizers.en.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # So, recalculate them outside the loop.
        self.transformer([encoder_input, output[:,:-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return text, tokens, attention_weights

    def print_translation(self, sentence, tokens, ground_truth):
        print(f'{"Input:":15s}: {sentence}')
        print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
        print(f'{"Ground truth":15s}: {ground_truth}')
# ===   === #


def main():
    # dataset params
    MAX_TOKENS      = 128
    BATCH_SIZE      = 64
    BUFFER_SIZE     = 20000

    # hyperparams
    NUM_LAYERS      = 4
    D_MODEL         = 128
    NUM_HEADS       = 8
    DFF             = 512
    DROPOUT_RATE    = 0.1

    # optimizer
    WARMUP_STEPS    = 4000
    BETA_1          = 0.9
    BETA_2          = 0.98
    EPSILON         = 1e-9

    # training params
    EPOCHS          = 20

    # aquire dataset pt - en
    dataset = DatasetHandler(
        dataset_name="ted_hrlr_translate/pt_to_en",
        model_name="ted_hrlr_translate_pt_en_converter",
        max_tokens=MAX_TOKENS,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE
    )
    examples, metadata = dataset(batch_config=Default_PT_EN_BatchConfig)

    # setup encoder, decoder
    encoder = Encoder(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        dropout_rate=DROPOUT_RATE
    )
    decoder = Decoder(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        dropout_rate=DROPOUT_RATE
    )

    # config encoder, decoder
    encoder.set_pre_layer(
        pre_layer=PositionalEmbedding(
            vocab_size=dataset.tokenizer.pt.get_vocab_size().numpy(),
            d_model=D_MODEL
        )
    )
    decoder.set_pre_layer(
        pre_layer=PositionalEmbedding(
            vocab_size=dataset.tokenizer.en.get_vocab_size().numpy(),
            d_model=D_MODEL
        )
    )
    decoder.set_post_layer(
        post_layer=tf.keras.layers.Dense(
            dataset.tokenizer.en.get_vocab_size().numpy()
        )
    )

    # setup, config transformer
    transformer = Transformer()

    transformer.set_encoder(encoder=encoder)
    transformer.set_decoder(decoder=decoder)

    # setup training params
    optimizer = tf.keras.optimizers.Adam(
        DefaultSchedule(
            d_model=D_MODEL,
            warmup_steps=WARMUP_STEPS
        ),
        beta_1=BETA_1,
        beta_2=BETA_2,
        epsilon=EPSILON
    )

    # training
    transformer.compile(
        loss=Loss.masked,
        optimizer=optimizer,
        metrics=[Accuarcy.masked]
    )
    transformer.fit(
        dataset.make_batches(examples['train']),
        epochs=EPOCHS,
        validation_data=dataset.make_batches(examples['validation'])
    )

    # translator v1
    translator = Translator(tokenizers, transformer)


if __name__ == "__main__":
    main()

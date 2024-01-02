import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

TRAIN   = False
LOAD    = False

PATH_TO_MODEL = None

for arg_index in range(len(sys.argv)):
    match sys.argv[arg_index]:
        case "-p":
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        case "--print":
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

        case "-t":
            TRAIN = True
        case "--train":
            TRAIN = True

        case "-l":
            arg_index += 1
            LOAD = True
            PATH_TO_MODEL = sys.argv[arg_index]

        case "--load":
            arg_index += 1
            LOAD = True
            PATH_TO_MODEL = sys.argv[arg_index]


import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text

from transformer.__init__ import *
from dataset.__init__ import *
from training.__init__ import *


# === PRE LAYERS (for TRANSLATOR v1) === #
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            d_model,
            mask_zero=True
        )
        self.pos_encoding = self.positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[tf.newaxis, :tf.shape(x)[1], :]
        return x

    def positional_encoding(self, length, depth):
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


# === BATCH CONFIGURATION (for TRANSLATOR v1) === #
class DefaultBatchConfig():
    def __init__(self, first_lang, sec_lang, max_tokens):
        self.first_lang = first_lang
        self.sec_lang = sec_lang
        self.max_tokens = max_tokens

    def __call__(self, first_lang, sec_lang):
        first_lang = self.first_lang.tokenize(first_lang)
        first_lang = first_lang[:, :self.max_tokens]
        first_lang = first_lang.to_tensor()

        sec_lang = self.sec_lang.tokenize(sec_lang)
        sec_lang = sec_lang[:, :(self.max_tokens+1)]
        sec_lang_sentences = sec_lang[:, :-1].to_tensor()
        sec_lang_labels = sec_lang[:, 1:].to_tensor()

        return (first_lang, sec_lang_sentences), sec_lang_labels
# ===   === #


# === TRANSLATOR v1 === #
class Translator():
    def __init__(self, tokenizer, first_lang, sec_lang, transformer):
        self.tokenizer = tokenizer
        self.first_lang = first_lang
        self.sec_lang = sec_lang

        self.transformer = transformer

    def __call__(self, sentence, max_tokens):
        translation = Translation(
            tokenizer=self.tokenizer,
            first_lang=self.first_lang,
            sec_lang=self.sec_lang,
            transformer=self.transformer
        )
        translation(
            sentence=tf.constant(sentence),
            output_array=tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True),
            max_tokens=max_tokens
        )

        return translation

class Translation():
    def __init__(self, tokenizer, first_lang, sec_lang, transformer):
        self.__tokenizer = tokenizer
        self.__first_lang = first_lang
        self.__sec_lang = sec_lang

        self.__transformer = transformer

    def __call__(self, sentence, output_array, max_tokens):
        # empty string handling
        if len(sentence.shape) == 0: sentence = sentence[tf.newaxis]

        # tokenize the sentence sentence
        sentence = self.__first_lang.tokenize(sentence).to_tensor()

        # output start- and end-tokens
        start_end_token = self.__sec_lang.tokenize([''])[0]

        start_token = start_end_token[0][tf.newaxis]
        end_token = start_end_token[1][tf.newaxis]

        # insert start token into output array
        output_array = output_array.write(0, start_token)

        # generating output
        for token_index in range(max_tokens):
            predictions = self.__transformer(
                [sentence, tf.transpose(output_array.stack())],
                training=False
            )
            predicted_token = tf.argmax(
                predictions[:, -1:, :],
                axis=-1
            )

            # insert predicted token
            output_array = output_array.write(token_index + 1, predicted_token[0])

            # exit the loop -> end of output
            if predicted_token == end_token: break

        output = tf.transpose(output_array.stack())

        # post processing
        self.text = self.__tokenizer.en.detokenize(output)[0]
        self.tokens = self.__tokenizer.en.lookup(output)[0]

        # aquires attention weights
        self.__transformer([sentence, output[:,:-1]], training=False)

        self.weights = self.__transformer.decoder.decoder_layers[-1].cross_attention.last_scores
# ===   === #



def main():
    # dataset params
    MAX_TOKENS      = 128
    BATCH_SIZE      = 64
    BUFFER_SIZE     = 20000

    # transformer hyperparams
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
    EPOCHS          = 2
    STEPS_PER_EPOCH = 5

    # aquire dataset pt - en
    pt_en_dataset = DatasetHandler(
        dataset_name="ted_hrlr_translate/pt_to_en",
        model_name="ted_hrlr_translate_pt_en_converter",
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE
    )
    pt_en_examples, pt_en_metadata = pt_en_dataset()

    pt_en_batch_config = DefaultBatchConfig(
        first_lang=pt_en_dataset.tokenizer.pt,
        sec_lang=pt_en_dataset.tokenizer.en,
        max_tokens=MAX_TOKENS
    )

    if LOAD: # load model from file
        # TODO
        transformer = tf.saved_model.load(PATH_TO_MODEL)

    if not LOAD: # create model from scratch
        # setup transformer
        transformer = Transformer()

        # setup encoder, decoder
        pt_encoder = Encoder(
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dff=DFF,
            dropout_rate=DROPOUT_RATE
        )
        en_decoder = Decoder(
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dff=DFF,
            dropout_rate=DROPOUT_RATE
        )

        # config encoder, decoder
        pt_encoder.set_pre_layer(
            pre_layer=PositionalEmbedding(
                vocab_size=pt_en_dataset.tokenizer.pt.get_vocab_size().numpy(),
                d_model=D_MODEL
            )
        )
        en_decoder.set_pre_layer(
            pre_layer=PositionalEmbedding(
                vocab_size=pt_en_dataset.tokenizer.en.get_vocab_size().numpy(),
                d_model=D_MODEL
            )
        )
        en_decoder.set_post_layer(
            post_layer=tf.keras.layers.Dense(
                pt_en_dataset.tokenizer.en.get_vocab_size().numpy()
            )
        )

        # config transformer
        transformer.set_encoder(encoder=pt_encoder)
        transformer.set_decoder(decoder=en_decoder)

    # prints model information
    transformer.info(pt_en_dataset.make_batches(
        dataset=pt_en_examples['train'],
        batch_config=pt_en_batch_config
    ).take(1))

    if TRAIN: # train model from supplied dataset
        # setup training params
        pt_en_tranining_examples = pt_en_dataset.make_batches(
            dataset=pt_en_examples['train'],
            batch_config=pt_en_batch_config
        )
        pt_en_validation_examples = pt_en_dataset.make_batches(
            dataset=pt_en_examples['validation'],
            batch_config=pt_en_batch_config
        )

        optimizer = tf.keras.optimizers.Adam(
            DefaultSchedule(
                d_model=D_MODEL,
                warmup_steps=WARMUP_STEPS
            ),
            beta_1=BETA_1,
            beta_2=BETA_2,
            epsilon=EPSILON
        )

        callbacks = tf.keras.callbacks.CallbackList()
        callbacks.append(DefaultCallback(transformer))

        # train
        transformer.compile(
            loss=Loss.masked,
            optimizer=optimizer,
            metrics=[Accuarcy.masked]
        )
        transformer.fit(
            pt_en_tranining_examples,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_data=pt_en_validation_examples,
            callbacks=callbacks
        )

    # translator
    pt_en_translate = Translator(
        tokenizer=pt_en_dataset.tokenizer,
        first_lang=pt_en_dataset.tokenizer.pt,
        sec_lang=pt_en_dataset.tokenizer.en,

        transformer=transformer
    )

    sample_tranlation = pt_en_translate(
        sentence="este é um problema que temos que resolver.",
        max_tokens=MAX_TOKENS
    ) # >>> "this is a problem we have to solve ."

    print(sample_tranlation.text)
    print(sample_tranlation.tokens)
    print(sample_tranlation.weights)

if __name__ == "__main__":
    main()

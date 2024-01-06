import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

TRAIN   = False
LOAD    = False

PATH_TO_MODEL = None

for arg_index in range(len(sys.argv)):
    match sys.argv[arg_index]:
        # tensorflow debugging mode
        case "-tf-d":
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        case "--tf-debug":
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

        # debugging mode
        case "-d":
            os.environ['MIN_LOG_LEVEL'] = '0'
        case "--debug":
            os.environ['MIN_LOG_LEVEL'] = '0'

        # trainging mode
        case "-t":
            TRAIN = True
        case "--train":
            TRAIN = True

        # load element <arg[arg_index+1]>
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
@tf.keras.saving.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.length = 2048

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.d_model,
            mask_zero=True
        )
        self.pos_encoding = self.positional_encoding()

    def call(self, x):
        x = self.embedding(x)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[tf.newaxis, :tf.shape(x)[1], :]

        return x

    def positional_encoding(self):
        depth = self.d_model / 2
        depths = np.arange(depth)[np.newaxis, :] / depth
        angle_rads = np.arange(self.length)[:, np.newaxis] / (10000**depths)

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1
        )

        return tf.cast(pos_encoding, dtype=tf.float32)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def get_config(self):
        super().get_config()
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model
        }
# ===   === #


# === BATCH CONFIGURATION (for TRANSLATOR v1) === #
class DefaultBatchConfig():
    def __init__(self, max_tokens, p_lang_tokenizer, s_lang_tokenizer):
        self.max_tokens = max_tokens

        self.p_lang_tokenizer = p_lang_tokenizer # primary language tokenizer
        self.s_lang_tokenizer = s_lang_tokenizer # secondary language tokenizer

    def __call__(self, p_lang, s_lang):
        # tokenize the input
        p_lang = self.p_lang_tokenizer.tokenize(p_lang)
        s_lang = self.s_lang_tokenizer.tokenize(s_lang)

        # only allow max number of tokens
        p_lang = p_lang[:, :self.max_tokens]
        s_lang = s_lang[:, :self.max_tokens +1] # +1 for end token

        #
        p_lang = p_lang.to_tensor()
        s_lang_sentence = s_lang[:, :-1].to_tensor()

        # label
        s_lang_labels = s_lang[:, 1:].to_tensor()

        return (p_lang, s_lang_sentence), s_lang_labels
# ===   === #


# === TRANSLATOR v1 === #
class Translator():
    def __init__(self, max_tokens, p_lang_tokenizer, s_lang_tokenizer, tokenizer, transformer):
        self.max_tokens = max_tokens

        self.p_lang_tokenizer = p_lang_tokenizer
        self.s_lang_tokenizer = s_lang_tokenizer

        self.tokenizer = tokenizer
        self.transformer = transformer

    def __call__(self, p_lang):
        # init translation object
        translation = Translation(
            max_tokens=self.max_tokens,

            p_lang_tokenizer=self.p_lang_tokenizer,
            s_lang_tokenizer=self.s_lang_tokenizer,

            tokenizer=self.tokenizer,
            transformer=self.transformer
        )

        # translate the input
        translation(
            p_lang=tf.constant(p_lang),
            s_lang_array=tf.TensorArray(
                dtype=tf.int64,
                size=0,
                dynamic_size=True
            )
        )

        return translation

class Translation():
    def __init__(self, max_tokens, p_lang_tokenizer, s_lang_tokenizer, tokenizer, transformer):
        self.max_tokens = max_tokens

        self.p_lang_tokenizer = p_lang_tokenizer
        self.s_lang_tokenizer = s_lang_tokenizer

        self.tokenizer = tokenizer
        self.transformer = transformer

    def __call__(self, p_lang, s_lang_array):
        # empty string handling
        if len(p_lang.shape) == 0: p_lang = p_lang[tf.newaxis]

        # tokenize
        p_lang = self.p_lang_tokenizer.tokenize(p_lang).to_tensor()

        # output start and end token
        s_lang_start_token = self.s_lang_tokenizer.tokenize([""])[0][0][tf.newaxis]
        s_lang_end_token = self.s_lang_tokenizer.tokenize([""])[0][1][tf.newaxis]

        # write start token to output array
        s_lang_array = s_lang_array.write(0, s_lang_start_token)

        # writing the rest of the output
        for i in tf.range(self.max_tokens):
            # transpose array to tensor
            s_lang = tf.transpose(s_lang_array.stack())

            # prediciton
            s_lang_token_predictions = self.transformer(
                [p_lang, s_lang],
                training=False
            )
            s_lang_token_prediction = tf.argmax(
                s_lang_token_predictions[:, -1:, :],
                axis=-1
            )

            # write predicted token to output array
            s_lang_array = s_lang_array.write(i +1, s_lang_token_prediction[0])

            # exiting if end token is last token
            if s_lang_token_prediction == s_lang_end_token:
                break

        # transpose array to tensor
        s_lang = tf.transpose(s_lang_array.stack())

        # attention weights
        self.transformer([p_lang, s_lang[:,:-1]], training=False)

        # attributes
        self.text = self.s_lang_tokenizer.detokenize(s_lang)[0]
        self.tokens = self.s_lang_tokenizer.lookup(s_lang)[0]
        self.weights = self.transformer.decoder.decoder_layers[-1].cross_attention.last_scores
# ===   === #



def main():
    # === PARAMS === #
    ## dataset params
    MAX_TOKENS      = 128
    BATCH_SIZE      = 64
    BUFFER_SIZE     = 20000

    ## transformer hyperparams
    NUM_LAYERS      = 6
    D_MODEL         = 512
    NUM_HEADS       = 8
    DFF             = 2048
    DROPOUT_RATE    = 0.1

    ## optimizer
    WARMUP_STEPS    = 4000
    BETA_1          = 0.9
    BETA_2          = 0.98
    EPSILON         = 1e-9

    ## training params
    EPOCHS          = 80
    STEPS_PER_EPOCH = 1200
    # ===   === #


    # === DATA HANDLING === #
    # pt-en dataset
    DATASET_NAME = "ted_hrlr_translate/pt_to_en"
    MODEL_NAME = "ted_hrlr_translate_pt_en_converter"

    dataset = DatasetHandler(
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE
    )

    # calling dataset content
    examples, metadata = dataset()
    # ===   === #


    # === TRANSFORMER === #
    transformer = Transformer()

    ## load
    if LOAD:
        # loading the encoder model from file
        encoder = tf.keras.models.load_model(
            filepath="__models__/pt-test-encoder.keras",
            custom_objects={
            }
        )

        # loading the decoder model from file
        decoder = tf.keras.models.load_model(
            filepath="__models__/en-test-decoder.keras",
            custom_objects={
            }
        )


    ## not load
    if not LOAD:
        # pt encoder
        encoder = Encoder(
            encoder_name="pt-encoder",
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dff=DFF,
            dropout_rate=DROPOUT_RATE,
            entry_layer=tf.keras.saving.serialize_keras_object(
                PositionalEmbedding(
                    vocab_size=dataset.tokenizer.pt.get_vocab_size().numpy(),
                    d_model=D_MODEL
                )
            )
        )

        # en decoder
        decoder = Decoder(
            decoder_name="en-decoder",
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dff=DFF,
            dropout_rate=DROPOUT_RATE,
            entry_layer=tf.keras.saving.serialize_keras_object(
                PositionalEmbedding(
                    vocab_size=dataset.tokenizer.en.get_vocab_size().numpy(),
                    d_model=D_MODEL
                )
            ),
            exit_layer=tf.keras.saving.serialize_keras_object(
                tf.keras.layers.Dense(
                    dataset.tokenizer.en.get_vocab_size().numpy()
                )
            )
        )
    # ===   === #


    # === TRAINING SCHEDULE - TRAIN === #
    if TRAIN:
        # init defaut batch configurator
        batch_config = DefaultBatchConfig(
            max_tokens=MAX_TOKENS,

            p_lang_tokenizer=dataset.tokenizer.pt,
            s_lang_tokenizer=dataset.tokenizer.en
        )

        # trainging schedule
        trainging_schedule = [
            TrainingScheduleElement(
                encoder=encoder, decoder=decoder,

                training_data=dataset.make_batches(
                    dataset=examples['train'],
                    batch_config=batch_config
                ),
                validation_data=dataset.make_batches(
                    dataset=examples['validation'],
                    batch_config=batch_config
                ),

                epochs=EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH,
                callbacks=[   ]
            )
        ]

        # compiling for trainging
        transformer.compile(
            loss=Loss.masked,
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=DefaultOptimizerSchedule(
                    d_model=D_MODEL,
                    warmup_steps=WARMUP_STEPS
                ),
                beta_1=BETA_1,
                beta_2=BETA_2,
                epsilon=EPSILON
            ),
            metrics=[Accuarcy.masked]
        )

        transformer.train(
            training_schedule=trainging_schedule,
            callbacks=[   ]
        )
    # ===   === #

if __name__ == "__main__":
    main()

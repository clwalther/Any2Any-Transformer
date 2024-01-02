import tensorflow as tf

from .encoder import Encoder
from .decoder import Decoder

class Transformer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def set_encoder(self, encoder: Encoder):
        self.encoder = encoder

    def set_decoder(self, decoder: Decoder):
        self.decoder = decoder

    def call(self, inputs):
        y, x  = inputs

        y = self.encoder(y)
        x = self.decoder(x, y)

        return x

    def info(self, build_example):
        for (pt, en), en_labels in build_example:
            break

        # builds the transformer with input
        self((pt, en))

        # print model summary
        print()
        self.summary()
        print()

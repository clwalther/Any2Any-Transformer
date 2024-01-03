import tensorflow as tf

from .encoder import Encoder
from .decoder import Decoder

class Transformer(tf.keras.Model):
    def __init__(self,*, num_layers, d_model, num_heads,
                    dff, dropout_rate):
        super().__init__()
        self.num_layers     = num_layers
        self.d_model        = d_model
        self.num_heads      = num_heads
        self.dff            = dff
        self.dropout_rate   = dropout_rate

        self.encoders       = list()
        self.decoders       = list()

        self.active_encoder = None
        self.active_decoder = None

    def call(self, inputs, active_encoder=None, active_decoder=None):
        not_none = lambda arg, s_arg: s_arg if arg is None else arg

        y, x  = inputs

        y = self.encoders[not_none(active_encoder, self.active_encoder)](y)
        x = self.decoders[not_none(active_decoder, self.active_decoder)](x, y)

        return x

    def train(self, training_schedule, callbacks=None):
        for element in training_schedule:
            # prints element
            element.get_information()

            self.active_encoder = element.active_encoder
            self.active_decoder = element.active_decoder

            # fitting with element
            self.fit(
                x=element.training_data,
                epochs=element.epochs,
                steps_per_epoch=element.steps_per_epoch,
                validation_data=element.validation_data,
                callbacks=element.callbacks
            )

            # run end of element callback
            if callbacks is not None:
                for callback in callbacks: callback()

    def add_encoder(self, name, pre_layer, num_layers=None, d_model=None,
                    num_heads=None, dff=None, dropout_rate=None):
        not_none = lambda arg, s_arg: s_arg if arg is None else arg

        self.encoders.append(
            Encoder(
                layer_name=name,
                num_layers=not_none(num_layers, self.num_layers),
                d_model=not_none(d_model, self.d_model),
                num_heads=not_none(num_heads, self.num_heads),
                dff=not_none(dff, self.dff),
                dropout_rate=not_none(dropout_rate, self.dropout_rate),
                pre_layer=pre_layer
            )
        )

        return len(self.encoders) - 1

    def add_decoder(self, name, pre_layer, post_layer, num_layers=None,
                    d_model=None, num_heads=None, dff=None, dropout_rate=None):
        not_none = lambda arg, s_arg: s_arg if arg is None else arg

        self.decoders.append(
            Decoder(
                layer_name=name,
                num_layers=not_none(num_layers, self.num_layers),
                d_model=not_none(d_model, self.d_model),
                num_heads=not_none(num_heads, self.num_heads),
                dff=not_none(dff, self.dff),
                dropout_rate=not_none(dropout_rate, self.dropout_rate),
                pre_layer=pre_layer,
                post_layer=post_layer
            )
        )

        return len(self.decoders) - 1

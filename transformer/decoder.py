import tensorflow as tf

from .attention import SelfAttention, CrossAttention
from .feedforward import FeedForward

@tf.keras.saving.register_keras_serializable()
class Decoder(tf.keras.Model):
    def __init__(self,*, decoder_name, num_layers, d_model, num_heads,
                    dff, dropout_rate, entry_layer, exit_layer):
        super().__init__()
        self.decoder_name = decoder_name
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.entry_layer = tf.keras.saving.deserialize_keras_object(entry_layer)
        self.exit_layer = tf.keras.saving.deserialize_keras_object(exit_layer)

        self.decoder_layers = [
            DecoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate
            )
        for _ in range(self.num_layers)]

        self.trainable = True

    def call(self, inputs):
        y, x = inputs

        x = self.entry_layer(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer((y, x))

        x = self.exit_layer(x)

        try:
            del x._keras_mask
        except AttributeError:
            pass

        return x

    def save(self, filepath, save_format=None):
        tf.keras.models.save_model(
            model=self,
            filepath=filepath,
            save_format=save_format
        )

    def get_config(self):
        super().get_config()
        return {
            "decoder_name": self.decoder_name,
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
            "entry_layer": self.entry_layer,
            "exit_layer": self.exit_layer
        }


@tf.keras.saving.register_keras_serializable()
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.self_attention = SelfAttention(
            casual_mask=True,
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.feedforward = FeedForward(d_model, dff, dropout_rate)

    def call(self, inputs):
        y, x = inputs

        x = self.self_attention(x=x)
        x = self.cross_attention(x=x, y=y)
        x = self.feedforward(x=x)

        return x

    def get_config(self):
        super().get_config()
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        }

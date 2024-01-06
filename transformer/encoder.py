import tensorflow as tf

from .attention import SelfAttention
from .feedforward import FeedForward

@tf.keras.saving.register_keras_serializable()
class Encoder(tf.keras.Model):
    def __init__(self,*, encoder_name, num_layers, d_model, num_heads,
                    dff, dropout_rate, entry_layer):
        super().__init__()
        self.encoder_name = encoder_name
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.entry_layer = tf.keras.saving.deserialize_keras_object(entry_layer)

        self.encoder_layers = [
            EncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate
            )
        for _ in range(self.num_layers)]

        self.trainable = True

    def call(self, x):
        x = self.entry_layer(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

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
            "encoder_name": self.encoder_name,
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
            "entry_layer": self.entry_layer
        }


@tf.keras.saving.register_keras_serializable()
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.self_attention = SelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.feedforward = FeedForward(d_model, dff, dropout_rate)

    def call(self, x):
        x = self.self_attention(x=x)
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

import tensorflow as tf

from .attention import SelfAttention
from .feedforward import FeedForward

class Encoder(tf.keras.layers.Layer):
    def __init__(self,*, num_layers, d_model, num_heads,
                         dff, dropout_rate):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.encoder_layers = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate
            )
        for encoder_layer_index in range(num_layers)]

    def set_pre_layer(self, pre_layer):
        self.pre_layer = pre_layer

    def call(self, x):
        x = self.pre_layer(x)
        x = self.dropout(x)

        for encoder_layer_index in range(self.num_layers):
            x = self.encoder_layers[encoder_layer_index](x)

        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate):
        super().__init__()

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

import tensorflow as tf

from .attention import BaseAttention, SelfAttention, CrossAttention
from .feedforward import FeedForward

class Decoder(tf.keras.layers.Layer):
    def __init__(self,*, num_layers, d_model, num_heads,
                         dff, dropout_rate):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.decoder_layers = [
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate
            )
        for decoder_layer_index in range(self.num_layers)]

    def set_pre_layer(self, pre_layer):
        self.pre_layer = pre_layer

    def set_post_layer(self, post_layer):
        self.post_layer = post_layer

    def call(self, x, y):
        x = self.pre_layer(x)
        x = self.dropout(x)

        for decoder_layer_index in range(self.num_layers):
            x = self.decoder_layers[decoder_layer_index](x, y)

        # final linear layer output
        x = self.post_layer(x)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del x._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate):
        super().__init__()

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

    def call(self, x, y):
        x = self.self_attention(x=x)
        x = self.cross_attention(x=x, y=y)
        x = self.feedforward(x=x)

        return x

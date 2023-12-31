import tensorflow as tf

from .attention import BaseAttention, SelfAttention, CrossAttention
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
            x = self.decoder_layers[encoder_layer_index](x, y)

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

def main():
    # en =
    # pt =
    # en_emb =
    # pt_emb =

    # === TESTING class: EncoderLayers ===
    sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)

    print(pt_emb.shape)                         # >>> (64, 62, 512)
    print(sample_encoder_layer(pt_emb).shape)   # >>> (64, 62, 512)

    # === TESTING class: Encoder ===
    sample_encoder = Encoder(
        num_layers=4,
        d_model=512,
        num_heads=8,
        dff=2048,
        vocab_size=8500
    )

    print(pt.shape)                                 # >>> (64, 62)
    print(sample_encoder(pt, training=False).shape) # >>> (64, 62, 512)


if __name__ == "__main__":
    main()

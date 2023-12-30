import tensorflow as tf

from .attention import BaseAttention, SelfAttention, CrossAttention
from .feedforward import FeedForward

class Decoder(tf.keras.layers.Layer):
    def __init__(self,*, num_layers, d_model, num_heads,
                         dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.decoder_layers = [
            DecoderLayers(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate
            )
        for decoder_layer_index in range(self.num_layers)]

    def call(self, x, y):
        x = self.dropout(x)

        for decoder_layer_index in range(self.num_layers):
            x = self.decoder_layers[decoder_layer_index](x, y)

        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = SelfAttention(
            casual_mask=True,
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.cross_attention = CrossAttention()
        self.feedforward = FeedForward(d_moderl, dff)

    def call(self, x, y):
        x = self.self_attention(x=x)
        x = self.cross_attention(x=x, y=y)
        x = self.feedforward(x=x)

        return x

def main():
    # en =
    # pt =
    # en_emb =
    # pt_emb =

    # === TESTING class: DecoderLayers ===
    sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048)

    # print shapes
    print(en_emb.shape)                                     # >>> (64, 58, 512)
    print(pt_emb.shape)                                     # >>> (64, 62, 512)
    print(sample_decoder_layer(x=en_emb, y=pt_emb).shape)   # >>> (64, 58, 512)

    # === TESTING class: Decoder ===
    sample_decoder = Decoder(
        num_layers=4,
        d_model=512,
        num_heads=8,
        dff=2048,
        vocab_size=8000
    )

    print(en.shape)                             # >>> (64, 58)
    print(pt_emb.shape)                         # >>> (64, 62, 512)
    print(sample_decoder(x=en, y=pt_emb).shape) # >>> (64, 58, 512)

    # (batch, heads, target_seq, input_seq)
    print(sample_decoder.decoder_layers[-1].cross_attention.last_scores.shape) # >>> TensorShape([64, 8, 58, 62])

if __name__ == "__main__":
    main()

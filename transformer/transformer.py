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

def main():
    # tokenizers =

    # === TESTING class: Transformer ===
    sample_transformer = Transformer(
        num_layers=5,
        d_model=128,
        num_heads=8,
        dff=512,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        dropout_rate=0.1
    )

    print(en.shape)                             # >>> (64, 58)
    print(pt.shape)                             # >>> (64, 62)
    print(sample_transformer((pt, en)).shape)   # >>> (64, 58, 7010)

    # (batch, heads, target_seq, input_seq)
    print(sample_transformer.decoder.decoder_layers[-1].cross_attention.last_scores.shape)  # >>> (64, 8, 58, 62)

    # print model summary
    sample_transformer.summary() # vvvvvvv

    # _________________________________________________________________
    # Layer (type)                Output Shape              Param #
    # =================================================================
    # encoder_1 (Encoder)         multiple                  3632768
    #
    # decoder_1 (Decoder)         multiple                  5647104
    #
    # dense_38 (Dense)            multiple                  904290
    #
    # =================================================================
    # Total params: 10184162 (38.85 MB)
    # Trainable params: 10184162 (38.85 MB)
    # Non-trainable params: 0 (0.00 Byte)
    # _________________________________________________________________

if __name__ == "__main__":
    main()

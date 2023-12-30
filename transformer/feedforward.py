import tensorflow as tf

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        # Feed Forward Network block
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        # Add & Norm block
        self.normalization = tf.keras.layers.LayerNormalization()
        self.addition = tf.keras.layers.Add()

    def call(self, x):
        output = self.network(x)

        x = self.addition([x, output])
        x = self.normalization(x)

        return x

def main():
    # en =
    # pt =
    # en_emb =
    # pt_emb =

    # === TESTING class: FeedForward ===
    sample_feed_forward_network = FeedForward(d_model=512, dff=2048)

    print(en_emb.shape)                                 # >>> (64, 58, 512)
    print(sample_feed_forward_network(en_emb).shape)    # >>> (64, 58, 512)


if __name__ == "__main__":
    main()
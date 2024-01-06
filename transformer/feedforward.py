import tensorflow as tf

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate

        ## Feed Forward Network block
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        ## Add & Norm block
        self.normalization = tf.keras.layers.LayerNormalization()
        self.addition = tf.keras.layers.Add()

    def call(self, x):
        output = self.network(x)

        x = self.addition([x, output])
        x = self.normalization(x)

        return x

    def get_config(self):
        super().get_config()
        return {
            "d_model": self.d_model,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        }

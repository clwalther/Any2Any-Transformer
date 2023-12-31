import tensorflow as tf

@tf.keras.saving.register_keras_serializable()
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, casual_mask=False, **kwargs):
        super().__init__()
        # public
        self.casual_mask = casual_mask

        ## Multi-Head attention block
        self.mulit_head_attention = tf.keras.layers.MultiHeadAttention(**kwargs)

        ## Add & Norm block
        self.normalization = tf.keras.layers.LayerNormalization()
        self.addition = tf.keras.layers.Add()

    def get_config(self):
        super().get_config()
        return {
            "casual_mask": self.casual_mask
        }


@tf.keras.saving.register_keras_serializable()
class SelfAttention(BaseAttention):
    def call(self, x):
        output = self.mulit_head_attention(
            query=x,
            value=x,
            key=x,
            use_causal_mask=self.casual_mask
        )

        x = self.addition([x, output])
        x = self.normalization(x)

        return x

@tf.keras.saving.register_keras_serializable()
class CrossAttention(BaseAttention):
    def call(self, x, y):
        output, scores = self.mulit_head_attention(
            query=x,
            key=y,
            value=y,
            return_attention_scores=True
        )

        # cache attention scores for later plotting
        self.last_scores = scores

        x = self.addition([x, output])
        x = self.normalization(x)

        return x

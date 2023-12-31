import tensorflow as tf

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, casual_mask=False, **kwargs):
        super().__init__()
        # Multi-Head attention block
        self.mulit_head_attention = tf.keras.layers.MultiHeadAttention(**kwargs)

        # Add & Norm block
        self.normalization = tf.keras.layers.LayerNormalization()
        self.addition = tf.keras.layers.Add()

class SelfAttention(BaseAttention):
    def call(self, x):
        output = self.mulit_head_attention(
            query=x,
            value=x,
            key=x,
            use_causal_mask=casual_mask
        )

        x = self.addition([x, output])
        x = self.normalization(x)

        return x

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

def main():
    # en =
    # pt =
    # en_emb =
    # pt_emb =

    # === TESTING class: SelfAttention (unmasked) ===
    sample_self_attention = SelfAttention(num_heads=2, key_dim=512)

    print(pt_emb.shape)                         # >>> (64, 62, 512)
    print(sample_self_attention(pt_emb).shape)  # >>> (64, 62, 512)

    # === TESTING class SelfAttention (masked) ===
    sample_self_attention = SelfAttention(casual_mask=True, num_heads=2, key_dim=512)

    print(pt_emb.shape)                         # >>> (64, 58, 512)
    print(sample_self_attention(pt_emb).shape)  # >>> (64, 58, 512)

    # === TESTING class: CrossAttention ===
    sample_cross_attention = CrossAttention(num_heads=2, key_dim=512)

    print(pt_emb.shape)                                 # >>> (64, 62, 512)
    print(en_emb.shape)                                 # >>> (64, 58, 512)
    print(sample_cross_attention(en_emb, pt_emb).shape) # >>> (64, 58, 512)

if __name__ == "__main__":
    main()

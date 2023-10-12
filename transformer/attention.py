import numpy as np


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = np.matmul(q, k.T)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = np.matmul(attention_weights, v)
    return output, attention_weights

def split_heads(x, d_model, num_heads):
    batch_size = tf.shape(x)[0]
    x = tf.reshape(x, (batch_size, -1, num_heads, d_model // num_heads))
    return np.transpose(x, perm=[0, 2, 1, 3])

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = split_heads(q, self.d_model, self.num_heads)
        k = split_heads(k, self.d_model, self.num_heads)
        v = split_heads(v, self.d_model, self.num_heads)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = np.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights

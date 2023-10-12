import numpy as np


def create_patches(image, patch_size):
    h, w, c = image.shape
    image = np.reshape(image, (h // patch_size, patch_size, w // patch_size, patch_size, c))
    image = np.transpose(image, (0, 2, 1, 3, 4))
    patches = np.reshape(image, (-1, patch_size, patch_size, c))
    return patches


def create_positional_encoding(length, d_model):
    pos = np.arange(length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    positional_encoding = np.zeros((length, d_model))
    positional_encoding[:, 0::2] = np.sin(pos * div_term)
    positional_encoding[:, 1::2] = np.cos(pos * div_term)
    return positional_encoding


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def multi_head_self_attention(q, k, v, num_heads):
    d_model = q.shape[-1]
    assert d_model % num_heads == 0
    depth = d_model // num_heads
    q = np.reshape(q, (q.shape[0], q.shape[1], num_heads, depth))
    k = np.reshape(k, (k.shape[0], k.shape[1], num_heads, depth))
    v = np.reshape(v, (v.shape[0], v.shape[1], num_heads, depth))
    scores = np.matmul(q, k.transpose((0, 1, 3, 2))) / np.sqrt(depth)
    attention_weights = softmax(scores)
    output = np.matmul(attention_weights, v)
    output = np.reshape(output, (output.shape[0], output.shape[1], d_model))
    return output


def feed_forward_network(x, dff):
    return np.dot(x, dff)


def layer_norm(x):
    return (x - np.mean(x)) / np.std(x)


def transformer_encoder(x, num_heads, dff):
    attention_output = multi_head_self_attention(x, x, x, num_heads)
    attention_output = layer_norm(attention_output + x)
    ffn_output = feed_forward_network(attention_output, dff)
    ffn_output = layer_norm(ffn_output + attention_output)
    return ffn_output

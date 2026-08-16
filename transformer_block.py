import numpy as np

from attention import MultiHeadAttention


def relu(x):
    return np.maximum(0, x)


class LayerNorm:
    def __init__(self, embed_dim, eps=1e-5):
        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.std_inv = 1 / np.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mean) * self.std_inv
        return self.gamma * self.x_hat + self.beta

    def backward(self, d_output, learning_rate=0.1):
        N = self.x.shape[-1]
        x_mu = self.x - self.mean

        d_x_hat = d_output * self.gamma
        d_var = np.sum(d_x_hat * x_mu * -0.5 * self.std_inv ** 3, axis=-1, keepdims=True)
        d_mean = np.sum(d_x_hat * -self.std_inv, axis=-1, keepdims=True) + \
            d_var * np.mean(-2 * x_mu, axis=-1, keepdims=True)
        d_x = d_x_hat * self.std_inv + d_var * 2 * x_mu / N + d_mean / N

        d_gamma = np.sum(d_output * self.x_hat, axis=0)
        d_beta = np.sum(d_output, axis=0)

        self.gamma -= learning_rate * d_gamma
        self.beta -= learning_rate * d_beta

        return d_x


class FeedForward:
    def __init__(self, embed_dim, hidden_dim):
        self.W1 = np.random.randn(embed_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, embed_dim) * 0.1
        self.b2 = np.zeros(embed_dim)

    def forward(self, x):
        self.x = x
        self.hidden_raw = x @ self.W1 + self.b1
        self.hidden = relu(self.hidden_raw)
        return self.hidden @ self.W2 + self.b2

    def backward(self, d_output, learning_rate=0.1):
        dW2 = self.hidden.T @ d_output
        db2 = d_output.sum(axis=0)

        d_hidden = d_output @ self.W2.T
        d_hidden_raw = d_hidden * (self.hidden_raw > 0)

        dW1 = self.x.T @ d_hidden_raw
        db1 = d_hidden_raw.sum(axis=0)
        d_x = d_hidden_raw @ self.W1.T

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        return d_x


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, ff_hidden_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x, causal=False):
        attn_out = self.attention.forward(x, causal=causal)
        x = self.norm1.forward(x + attn_out)

        ff_out = self.feed_forward.forward(x)
        x = self.norm2.forward(x + ff_out)

        return x

    def backward(self, d_output, learning_rate=0.1):
        d_x2 = self.norm2.backward(d_output, learning_rate)
        d_ff_out = d_x2
        d_n1_via_ff = self.feed_forward.backward(d_ff_out, learning_rate)
        d_n1_total = d_x2 + d_n1_via_ff

        d_x1 = self.norm1.backward(d_n1_total, learning_rate)
        d_attn_out = d_x1
        d_x_via_attn = self.attention.backward(d_attn_out, learning_rate)
        d_x_total = d_x1 + d_x_via_attn

        return d_x_total


if __name__ == "__main__":
    from embeddings import Tokenizer, Embedding
    from positional_encoding import get_positional_encoding

    sentence = "the animal did not cross the street because it was tired"
    tok = Tokenizer(sentence)
    words = sentence.split()
    token_ids = tok.encode(sentence)

    embed_dim = 8
    embedding = Embedding(tok.vocab_size, embed_dim)
    token_vectors = embedding.forward(token_ids)
    pos_encoding = get_positional_encoding(len(words), embed_dim)

    block_input = token_vectors + pos_encoding

    print(f"Sentence: '{sentence}'")
    print(f"Input shape going into the transformer block: {block_input.shape}")

    block = TransformerBlock(embed_dim, num_heads=4, ff_hidden_dim=16)
    output = block.forward(block_input)

    print(f"Output shape coming out: {output.shape}")
    print("Checking LayerNorm worked (should be ~0 mean, ~1 variance per word):")
    for word, vec in zip(words[:3], output[:3]):
        print(f"  '{word}': mean={vec.mean():.4f}, var={vec.var():.4f}")
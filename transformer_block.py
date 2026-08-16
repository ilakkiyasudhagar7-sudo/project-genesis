import numpy as np

from attention import MultiHeadAttention


def relu(x):
    return np.maximum(0, x)


class LayerNorm:
    """
    Rescales each word's vector to mean 0, variance 1, then applies a
    learned scale (gamma) and shift (beta) so the model can undo the
    normalization if that's ever actually useful.

    Keeps numbers stable as they flow through many stacked blocks —
    without this, values tend to explode or vanish across depth.
    """

    def __init__(self, embed_dim, eps=1e-5):
        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta


class FeedForward:
    """
    A small 2-layer network applied to EVERY word's vector independently
    (same weights reused at every position). Expands to a wider hidden
    size, applies ReLU, then projects back down — this is where the
    model does most of its actual "thinking" about each word's content,
    as opposed to attention, which is about relationships BETWEEN words.
    """

    def __init__(self, embed_dim, hidden_dim):
        self.W1 = np.random.randn(embed_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, embed_dim) * 0.1
        self.b2 = np.zeros(embed_dim)

    def forward(self, x):
        hidden = relu(x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


class TransformerBlock:
    """
    One full transformer block: multi-head attention -> residual + norm
    -> feedforward -> residual + norm.

    The residual connections (x + sublayer(x)) mean each piece only has
    to learn a CORRECTION to add on top of its input, not replace it
    entirely — this is what makes stacking many of these blocks
    trainable instead of degrading. Real GPTs are just many of these
    blocks stacked on top of each other (Stage 9).
    """

    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, ff_hidden_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x):
        # residual connection around attention: add its output back onto the input
        attn_out = self.attention.forward(x)
        x = self.norm1.forward(x + attn_out)

        # residual connection around feedforward: same pattern
        ff_out = self.feed_forward.forward(x)
        x = self.norm2.forward(x + ff_out)

        return x


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

    # this is the full pipeline so far: tokenize -> embed -> add position
    block_input = token_vectors + pos_encoding

    print(f"Sentence: '{sentence}'")
    print(f"Input shape going into the transformer block: {block_input.shape}")

    block = TransformerBlock(embed_dim, num_heads=4, ff_hidden_dim=16)
    output = block.forward(block_input)

    print(f"Output shape coming out: {output.shape}  <- MUST match input shape,")
    print("since blocks get stacked (Stage 9) and each one's output feeds the next.\n")

    # prove LayerNorm actually did its job: mean ~0, variance ~1 per word
    print("Checking LayerNorm worked (should be ~0 mean, ~1 variance per word):")
    for word, vec in zip(words[:3], output[:3]):
        print(f"  '{word}': mean={vec.mean():.4f}, var={vec.var():.4f}")

    print("\nA real model stacks many of these blocks back to back —")
    print("that's literally Stage 9.")
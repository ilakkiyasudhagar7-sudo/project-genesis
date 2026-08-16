import numpy as np


def softmax(x):
    # subtract max for numerical stability (prevents overflow in exp)
    # doesn't change the result: softmax is shift-invariant
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class SelfAttention:
    """
    For every word: build a Query (what am I looking for?), a Key (what do
    I contain?), and a Value (what do I actually offer?) using three
    separate learned weight matrices.

    Then: compare every word's Query against every OTHER word's Key to get
    attention scores, turn those into percentages (softmax), and use those
    percentages to blend together everyone's Values. That blend becomes
    the new, context-aware vector for each word.
    """

    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        # one weight matrix per Q/K/V, each mapping embed_dim -> embed_dim
        self.W_query = np.random.randn(embed_dim, embed_dim) * 0.1
        self.W_key = np.random.randn(embed_dim, embed_dim) * 0.1
        self.W_value = np.random.randn(embed_dim, embed_dim) * 0.1

    def forward(self, embeddings):
        """
        embeddings: shape (seq_len, embed_dim) — one vector per word,
        e.g. straight out of the Embedding lookup from Stage 4.
        """
        self.embeddings = embeddings

        # project every word's embedding into its Query, Key, Value
        Q = embeddings @ self.W_query   # (seq_len, embed_dim)
        K = embeddings @ self.W_key     # (seq_len, embed_dim)
        V = embeddings @ self.W_value   # (seq_len, embed_dim)

        # every word's Query compared against every word's Key
        # (seq_len, embed_dim) @ (embed_dim, seq_len) -> (seq_len, seq_len)
        # scores[i][j] = how much word i should attend to word j
        scores = Q @ K.T

        # scale down before softmax — without this, large embed_dims push
        # scores to extremes and softmax gets nearly one-hot (gradients vanish)
        scores = scores / np.sqrt(self.embed_dim)

        # turn each row of raw scores into percentages that sum to 1
        self.attention_weights = softmax(scores)

        # blend every word's Value according to those percentages
        output = self.attention_weights @ V   # (seq_len, embed_dim)

        return output


class MultiHeadAttention:
    """
    Run several independent SelfAttention computations in parallel — each
    a separate "head" with its own Q/K/V weights, operating on a SLICE of
    the embedding (embed_dim split evenly across heads). Different heads
    can end up specializing in different kinds of relationships (grammar,
    meaning, position...) once trained.

    Outputs from all heads are concatenated back to embed_dim, then mixed
    together with one more learned matrix, W_output.
    """

    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0, "embed_dim must divide evenly by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # one independent SelfAttention "head" per split of the embedding
        self.heads = [SelfAttention(self.head_dim) for _ in range(num_heads)]

        # mixes the concatenated heads back into one embed_dim-sized vector
        self.W_output = np.random.randn(embed_dim, embed_dim) * 0.1

    def forward(self, embeddings):
        # split each word's embedding into num_heads equal chunks
        # e.g. embed_dim=8, num_heads=2 -> each head sees a (seq_len, 4) slice
        chunks = np.split(embeddings, self.num_heads, axis=-1)

        # run every head's attention independently on its own chunk
        head_outputs = [head.forward(chunk) for head, chunk in zip(self.heads, chunks)]

        # stitch the heads' outputs back together side by side
        concatenated = np.concatenate(head_outputs, axis=-1)   # (seq_len, embed_dim)

        # one more learned mix so heads can combine information, not just sit side by side
        return concatenated @ self.W_output


if __name__ == "__main__":
    from embeddings import Tokenizer, Embedding

    sentence = "the animal did not cross the street because it was tired"
    tok = Tokenizer(sentence)
    token_ids = tok.encode(sentence)
    words = sentence.split()

    embed_dim = 8
    embedding = Embedding(tok.vocab_size, embed_dim)
    vectors = embedding.forward(token_ids)   # (seq_len, embed_dim)

    attention = SelfAttention(embed_dim)
    output = attention.forward(vectors)

    print(f"Sentence: '{sentence}'")
    print(f"Words: {words}\n")

    # find "it" and print who it's attending to, most-to-least
    it_index = words.index("it")
    it_weights = attention.attention_weights[it_index]

    print("Attention from 'it' to every other word (single head):")
    ranked = sorted(zip(words, it_weights), key=lambda pair: -pair[1])
    for word, weight in ranked:
        bar = "#" * int(weight * 50)
        print(f"  {word:10s} {weight:.4f} {bar}")

    print(f"\n(Weights sum to {it_weights.sum():.4f} — softmax always sums to 1)")
    print("\nNote: weights are random/untrained right now, so this ranking")
    print("isn't meaningful yet — it just proves the MECHANISM works.")
    print("Real 'it' -> 'animal' understanding only emerges after training")
    print("on lots of text (Stage 9), same as embeddings.")

    # --- Stage 6: same sentence, but through multiple heads at once ---
    print("\n\n--- Multi-head attention (4 heads) ---")
    num_heads = 4
    mha = MultiHeadAttention(embed_dim, num_heads)
    mha_output = mha.forward(vectors)

    print(f"Output shape: {mha_output.shape} (still seq_len={len(words)} x embed_dim={embed_dim})")
    print("Each head's attention weights for 'it' (should differ slightly per head,")
    print("since each has independent random Q/K/V weights):\n")

    for h, head in enumerate(mha.heads):
        head_weights = head.attention_weights[it_index]
        rounded = np.round(head_weights, 4)
        print(f"  Head {h}: {rounded}")

    print("\nWith random, UNTRAINED weights these stay close to uniform (~1/11")
    print("each) and mostly look alike — that's expected, same as Stage 5.")
    print("The point of this demo isn't that these heads are already smart;")
    print("it's that each head runs its own independent Q/K/V computation,")
    print("which is the machinery that lets heads specialize ONCE TRAINED.")
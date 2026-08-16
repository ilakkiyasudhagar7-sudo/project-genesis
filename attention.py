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

    print("Attention from 'it' to every other word:")
    ranked = sorted(zip(words, it_weights), key=lambda pair: -pair[1])
    for word, weight in ranked:
        bar = "#" * int(weight * 50)
        print(f"  {word:10s} {weight:.4f} {bar}")

    print(f"\n(Weights sum to {it_weights.sum():.4f} — softmax always sums to 1)")
    print("\nNote: weights are random/untrained right now, so this ranking")
    print("isn't meaningful yet — it just proves the MECHANISM works.")
    print("Real 'it' -> 'animal' understanding only emerges after training")
    print("on lots of text (Stage 9), same as embeddings.")
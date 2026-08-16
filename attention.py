import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class SelfAttention:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.W_query = np.random.randn(embed_dim, embed_dim) * 0.1
        self.W_key = np.random.randn(embed_dim, embed_dim) * 0.1
        self.W_value = np.random.randn(embed_dim, embed_dim) * 0.1

    def forward(self, embeddings, causal=False):
        self.embeddings = embeddings
        self.causal = causal

        Q = embeddings @ self.W_query
        K = embeddings @ self.W_key
        V = embeddings @ self.W_value
        self.Q, self.K, self.V = Q, K, V

        scores = Q @ K.T
        scores = scores / np.sqrt(self.embed_dim)

        if causal:
            seq_len = embeddings.shape[0]
            mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            scores = np.where(mask, -1e9, scores)

        self.attention_weights = softmax(scores)
        output = self.attention_weights @ V

        return output

    def backward(self, d_output, learning_rate=0.1):
        A = self.attention_weights

        dV = A.T @ d_output
        dA = d_output @ self.V.T

        dscores = A * (dA - np.sum(dA * A, axis=-1, keepdims=True))
        dscores = dscores / np.sqrt(self.embed_dim)

        dQ = dscores @ self.K
        dK = dscores.T @ self.Q

        dW_query = self.embeddings.T @ dQ
        dW_key = self.embeddings.T @ dK
        dW_value = self.embeddings.T @ dV

        d_input = dQ @ self.W_query.T + dK @ self.W_key.T + dV @ self.W_value.T

        self.W_query -= learning_rate * dW_query
        self.W_key -= learning_rate * dW_key
        self.W_value -= learning_rate * dW_value

        return d_input


class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0, "embed_dim must divide evenly by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.heads = [SelfAttention(self.head_dim) for _ in range(num_heads)]
        self.W_output = np.random.randn(embed_dim, embed_dim) * 0.1

    def forward(self, embeddings, causal=False):
        chunks = np.split(embeddings, self.num_heads, axis=-1)
        head_outputs = [head.forward(chunk, causal=causal) for head, chunk in zip(self.heads, chunks)]
        self.concatenated = np.concatenate(head_outputs, axis=-1)
        return self.concatenated @ self.W_output

    def backward(self, d_output, learning_rate=0.1):
        dW_output = self.concatenated.T @ d_output
        d_concatenated = d_output @ self.W_output.T

        d_chunks = np.split(d_concatenated, self.num_heads, axis=-1)
        d_input_chunks = [head.backward(d_chunk, learning_rate) for head, d_chunk in zip(self.heads, d_chunks)]

        self.W_output -= learning_rate * dW_output

        return np.concatenate(d_input_chunks, axis=-1)


if __name__ == "__main__":
    from embeddings import Tokenizer, Embedding

    sentence = "the animal did not cross the street because it was tired"
    tok = Tokenizer(sentence)
    token_ids = tok.encode(sentence)
    words = sentence.split()

    embed_dim = 8
    embedding = Embedding(tok.vocab_size, embed_dim)
    vectors = embedding.forward(token_ids)

    attention = SelfAttention(embed_dim)
    output = attention.forward(vectors)

    print(f"Sentence: '{sentence}'")
    print(f"Words: {words}\n")

    it_index = words.index("it")
    it_weights = attention.attention_weights[it_index]

    print("Attention from 'it' to every other word (single head):")
    ranked = sorted(zip(words, it_weights), key=lambda pair: -pair[1])
    for word, weight in ranked:
        bar = "#" * int(weight * 50)
        print(f"  {word:10s} {weight:.4f} {bar}")

    print(f"\n(Weights sum to {it_weights.sum():.4f})")

    print("\n\n--- Multi-head attention (4 heads) ---")
    num_heads = 4
    mha = MultiHeadAttention(embed_dim, num_heads)
    mha_output = mha.forward(vectors)

    print(f"Output shape: {mha_output.shape}")
    for h, head in enumerate(mha.heads):
        head_weights = head.attention_weights[it_index]
        rounded = np.round(head_weights, 4)
        print(f"  Head {h}: {rounded}")
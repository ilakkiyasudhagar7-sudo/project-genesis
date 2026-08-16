import numpy as np


class Embedding:
    """
    A lookup table: one row of numbers (a vector) per token in the vocabulary.
    weights shape: (vocab_size, embedding_dim)

    This is literally just a matrix. "Looking up" a word's embedding means
    grabbing the row at that word's index — no math beyond indexing.
    """

    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.01

    def forward(self, token_ids):
        # token_ids: a list of integers, e.g. [4, 1, 7]
        # returns one vector per token: shape (len(token_ids), embedding_dim)
        self.token_ids = np.array(token_ids)
        return self.weights[self.token_ids]

    def backward(self, d_output, learning_rate=0.1):
        """
        d_output: gradient of the loss w.r.t. each output vector,
        shape (len(token_ids), embedding_dim).

        Only the rows for tokens we actually USED get updated — every
        other word in the vocabulary is untouched this step. That's why
        training embeddings needs LOTS of text: each word only learns
        when it actually shows up.
        """
        for i, token_id in enumerate(self.token_ids):
            self.weights[token_id] -= learning_rate * d_output[i]


class Tokenizer:
    """Bare-minimum word-level tokenizer: splits on spaces, assigns each
    unique word an integer id. Real GPTs use sub-word tokenization
    (e.g. BPE) so they can handle any word, even ones never seen before —
    but the CONCEPT (word/piece -> integer id) is identical."""

    def __init__(self, text):
        words = text.lower().split()
        unique_words = sorted(set(words))
        self.word_to_id = {word: i for i, word in enumerate(unique_words)}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}

    def encode(self, text):
        return [self.word_to_id[word] for word in text.lower().split()]

    def decode(self, token_ids):
        return " ".join(self.id_to_word[i] for i in token_ids)

    @property
    def vocab_size(self):
        return len(self.word_to_id)


if __name__ == "__main__":
    corpus = "the cat sat on the mat the dog sat on the rug"

    tok = Tokenizer(corpus)
    print(f"Vocabulary ({tok.vocab_size} words): {tok.word_to_id}")

    sentence = "the cat sat"
    token_ids = tok.encode(sentence)
    print(f"\n'{sentence}' -> token ids: {token_ids}")

    embed_dim = 4
    embedding = Embedding(tok.vocab_size, embed_dim)
    vectors = embedding.forward(token_ids)

    print(f"\nEach word becomes a {embed_dim}-number vector:")
    for word, token_id, vec in zip(sentence.split(), token_ids, vectors):
        print(f"  '{word}' (id={token_id}) -> {np.round(vec, 4)}")

    # prove the lookup is consistent: same word -> same vector, every time
    repeat_ids = tok.encode("the")
    repeat_vec = embedding.forward(repeat_ids)[0]
    print(f"\n'the' looked up again -> {np.round(repeat_vec, 4)} (should match 'the' above)")

    print(f"\nDecoding ids {token_ids} back to words: '{tok.decode(token_ids)}'")
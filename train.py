import numpy as np

from embeddings import Tokenizer, Embedding
from positional_encoding import get_positional_encoding
from transformer_block import TransformerBlock


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class GPTModel:
    """
    Everything from Stages 4-8, stacked into one model:
    tokens -> embeddings -> + positional encoding -> N transformer blocks
    -> a final linear layer projecting each word's vector to vocab_size
    scores (logits), one score per possible next word.
    """

    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_blocks, max_seq_len):
        self.embedding = Embedding(vocab_size, embed_dim)
        # build a generously-sized table once; forward() slices what it needs,
        # so this works for training AND for generation (which grows the
        # sequence one token at a time, past the original training length)
        self.pos_encoding = get_positional_encoding(max_seq_len, embed_dim)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim)
            for _ in range(num_blocks)
        ]
        # final projection: embed_dim -> vocab_size, so we get one score
        # per vocabulary word at every position ("how likely is this word next?")
        self.W_out = np.random.randn(embed_dim, vocab_size) * 0.1
        self.b_out = np.zeros(vocab_size)

    def forward(self, token_ids):
        token_vectors = self.embedding.forward(token_ids)
        current_len = len(token_ids)
        self.block_input = token_vectors + self.pos_encoding[:current_len]

        x = self.block_input
        for block in self.blocks:
            x = block.forward(x, causal=True)   # causal: no peeking at future words
        self.final_hidden = x

        logits = x @ self.W_out + self.b_out    # (seq_len, vocab_size)
        return logits

    def backward(self, d_logits, learning_rate=0.1):
        dW_out = self.final_hidden.T @ d_logits
        db_out = d_logits.sum(axis=0)

        d_x = d_logits @ self.W_out.T
        for block in reversed(self.blocks):
            d_x = block.backward(d_x, learning_rate)

        # positional encoding has no learned params, and is just ADDED on,
        # so its gradient passes straight through unchanged to the embedding
        self.embedding.backward(d_x, learning_rate)

        self.W_out -= learning_rate * dW_out
        self.b_out -= learning_rate * db_out

    def train_step(self, input_ids, target_ids, learning_rate=0.1):
        logits = self.forward(input_ids)          # (seq_len, vocab_size)
        probs = softmax(logits)

        # cross-entropy loss: -log(probability assigned to the correct word)
        seq_len = len(target_ids)
        correct_probs = probs[np.arange(seq_len), target_ids]
        loss = -np.mean(np.log(correct_probs + 1e-9))

        # gradient of softmax + cross-entropy together: beautifully simple —
        # just (predicted probabilities - the correct answer as one-hot)
        d_logits = probs.copy()
        d_logits[np.arange(seq_len), target_ids] -= 1
        d_logits /= seq_len

        self.backward(d_logits, learning_rate)
        return loss

    def generate_next(self, token_ids):
        logits = self.forward(token_ids)
        probs = softmax(logits)
        return probs[-1]   # probability distribution for the NEXT word after the last one


if __name__ == "__main__":
    import re

    # public-domain text: Hamlet's "To be, or not to be" soliloquy —
    # real variety (73 unique words) instead of one repeated toy sentence
    raw_text = """To be, or not to be, that is the question:
    Whether tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die, to sleep,
    No more, and by a sleep to say we end
    The heartache and the thousand natural shocks
    That flesh is heir to, tis a consummation
    Devoutly to be wished. To die, to sleep,
    To sleep, perchance to dream, ay theres the rub,
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause, theres the respect
    That makes calamity of so long life."""

    # clean: lowercase, strip punctuation so "be," and "be" aren't different tokens
    corpus = re.sub(r'[^a-z\s]', '', raw_text.lower())
    corpus = re.sub(r'\s+', ' ', corpus).strip()

    tok = Tokenizer(corpus)
    words = corpus.split()
    token_ids = tok.encode(corpus)
    seq_len = len(token_ids) - 1   # last token has no "next word" to predict

    print(f"Vocabulary ({tok.vocab_size} words)")
    print(f"Training sequence length: {seq_len}\n")

    # next-token prediction: input is all-but-last, target is all-but-first
    # (shifted by one — this is THE training signal for a GPT-style model)
    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]

    model = GPTModel(
        vocab_size=tok.vocab_size,
        embed_dim=32,
        num_heads=4,
        ff_hidden_dim=64,
        num_blocks=2,
        max_seq_len=seq_len,
    )

    print("Training...")
    for epoch in range(800):
        loss = model.train_step(input_ids, target_ids, learning_rate=0.05)
        if epoch % 80 == 0:
            print(f"epoch {epoch:4d} | loss={loss:.4f}")

    print(f"epoch  800 | loss={loss:.4f} (final)")

    print("\n--- Checking what it learned ---")
    probs = softmax(model.forward(input_ids))
    correct = 0
    for i in range(seq_len):
        predicted_id = np.argmax(probs[i])
        predicted_word = tok.id_to_word[predicted_id]
        actual_word = tok.id_to_word[target_ids[i]]
        match = "correct" if predicted_id == target_ids[i] else "wrong"
        if predicted_id == target_ids[i]:
            correct += 1
        print(f"  after '{words[i]}' -> predicted '{predicted_word}' (actual: '{actual_word}') [{match}]")

    print(f"\nAccuracy on training data: {correct}/{seq_len}")
    print("\nWith 73 unique words instead of 7, this is a real step up in")
    print("difficulty from the toy demo — still memorizing at this scale,")
    print("but the model now has to tell apart far more genuine patterns.")
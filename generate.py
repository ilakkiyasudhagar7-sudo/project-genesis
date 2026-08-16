import numpy as np

from embeddings import Tokenizer
from train import GPTModel, softmax


def generate(model, tok, seed_text, max_new_tokens=8):
    """
    The actual payoff: feed some starting words in, predict the next word,
    APPEND it to the sequence, then predict again using the now-longer
    sequence — repeat. This is exactly how every GPT generates text.
    """
    token_ids = tok.encode(seed_text)

    for _ in range(max_new_tokens):
        probs = model.generate_next(token_ids)
        next_id = int(np.argmax(probs))   # greedy: always take the most likely word
        token_ids.append(next_id)

    return tok.decode(token_ids)


if __name__ == "__main__":
    import re

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

    corpus = re.sub(r'[^a-z\s]', '', raw_text.lower())
    corpus = re.sub(r'\s+', ' ', corpus).strip()

    tok = Tokenizer(corpus)
    token_ids = tok.encode(corpus)
    seq_len = len(token_ids) - 1

    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]

    model = GPTModel(
        vocab_size=tok.vocab_size,
        embed_dim=32,
        num_heads=4,
        ff_hidden_dim=64,
        num_blocks=2,
        max_seq_len=140,   # generously sized: training needs 116, generation grows further
    )

    print("Training (same as Stage 9)...")
    for epoch in range(800):
        loss = model.train_step(input_ids, target_ids, learning_rate=0.05)
    print(f"Final loss: {loss:.4f}\n")

    print("--- Generating text, one word at a time ---")
    seeds = ["to be", "the mind", "for in that sleep"]
    for seed in seeds:
        generated = generate(model, tok, seed, max_new_tokens=8)
        print(f"Seed: '{seed}'  ->  '{generated}'")

    print("\nStill memorizing rather than truly generalizing — 116 words of")
    print("training data isn't enough to generalize to genuinely NEW text,")
    print("just to reproduce/continue patterns from what it's seen. Real GPTs")
    print("need billions of words before genuinely novel generation emerges.")
    print("But the machinery — tokenize, embed, attend, predict, append,")
    print("repeat — is exactly what every real GPT does. You built it.")
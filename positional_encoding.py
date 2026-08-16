import numpy as np


def get_positional_encoding(seq_len, embed_dim):
    """
    Returns a (seq_len, embed_dim) matrix: one unique "position fingerprint"
    per position, built from sine/cosine waves at different frequencies.

    This gets ADDED directly onto token embeddings (not concatenated) so
    the model can tell WHERE each word is — attention itself has no sense
    of order, it only sees relationships, so without this "cat sat mat"
    and "mat sat cat" would look identical to it.
    """
    positions = np.arange(seq_len)[:, np.newaxis]   # (seq_len, 1)
    dims = np.arange(embed_dim)[np.newaxis, :]        # (1, embed_dim)

    # frequency shrinks across dimensions: early dims oscillate fast (fine-
    # grained position info), later dims oscillate slow (coarse position info)
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(embed_dim))
    angles = positions * angle_rates                  # (seq_len, embed_dim)

    pe = np.zeros((seq_len, embed_dim))
    pe[:, 0::2] = np.sin(angles[:, 0::2])   # even dimensions -> sine
    pe[:, 1::2] = np.cos(angles[:, 1::2])   # odd dimensions -> cosine
    return pe


if __name__ == "__main__":
    from embeddings import Tokenizer, Embedding

    sentence = "the cat sat"
    tok = Tokenizer(sentence)
    words = sentence.split()
    embed_dim = 6

    embedding = Embedding(tok.vocab_size, embed_dim)
    token_ids = tok.encode(sentence)
    token_vectors = embedding.forward(token_ids)          # (seq_len, embed_dim)

    pos_encoding = get_positional_encoding(len(words), embed_dim)

    print(f"Sentence: '{sentence}'\n")
    print("Positional fingerprint is different for every position,")
    print("regardless of which word sits there:")
    for i, word in enumerate(words):
        print(f"  pos {i} ('{word}'): {np.round(pos_encoding[i], 4)}")

    # the actual input to attention: token meaning + position, added together
    final_input = token_vectors + pos_encoding
    print("\nToken embedding + positional encoding = final input to attention:")
    for word, tok_vec, final in zip(words, token_vectors, final_input):
        print(f"  '{word}': token={np.round(tok_vec, 4)} + pos -> final={np.round(final, 4)}")

    # prove order now matters: same words, swapped order -> different vectors
    print("\n--- Proving order now matters ---")
    swapped = "cat sat the"
    swapped_words = swapped.split()
    swapped_ids = tok.encode(swapped)
    swapped_token_vectors = embedding.forward(swapped_ids)
    swapped_pos_encoding = get_positional_encoding(len(swapped_words), embed_dim)
    swapped_final = swapped_token_vectors + swapped_pos_encoding

    the_original_index = words.index("the")
    the_swapped_index = swapped_words.index("the")

    print(f"'the' at position {the_original_index} in '{sentence}':")
    print(f"  {np.round(final_input[the_original_index], 4)}")
    print(f"'the' at position {the_swapped_index} in '{swapped}':")
    print(f"  {np.round(swapped_final[the_swapped_index], 4)}")
    print("\nSame word, same learned embedding — but different final vectors,")
    print("because positional encoding is added on top based on WHERE it sits.")
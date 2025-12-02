"""
Example implementation of Byte Pair Encoding (BPE) tokenization.

This module demonstrates a basic BPE tokenizer that merges frequent byte pairs
to build a vocabulary and compress text sequences.
"""
import numpy as np
text = """
"""

tokens = text.encode("utf-8") # Raw bytes
tokens = list(map(int, tokens)) # converts to a list of ints in range 0 to 255 for convenience

print("---")
print(text)
print(f"length: {len(text)} code points")
print("---")
print(tokens)
print(f"length: {len(tokens)} bytes/tokens") # is higher because unicode complex characters become up to 4 bytes
print("---\n")


def get_stats(ids):
    """
    Given a list of ids, returns a dictionary of pair counts.
    The key is a tuple of two adjacent ids, and the value is the count of that pair.
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    Merge a pair of adjacent ids in a list of ids to a single idx.

    Args:
        ids (list): The list of ids to merge.
        pair (tuple): The pair of ids to merge.
        idx (int): The id to replace the pair with.

    Returns:
        list: The list of ids with the pair merged.
    """
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids [i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

vocab_size = 1000
num_merges = vocab_size - 256
ids = list(tokens)

merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256+i
    print(f"merging {pair} into a new token: {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0,p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    """
    Decode a list of token IDs back to text.

    Args:
        ids (list): List of token IDs to decode.

    Returns:
        str: Decoded text string.
    """
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors = 'replace')
    return text

# decoded_text = decode(ids)
# print(decoded_text)

def encode(text):
    """
    Encode text into a list of token IDs using learned BPE merges.

    Args:
        text (str): Input text to encode.

    Returns:
        list: List of token IDs.
    """
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key = lambda pair: merges.get(pair, float('inf')))
        if pair not in merges:
            break # nothing else to merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


print(encode("Hello, world!"))
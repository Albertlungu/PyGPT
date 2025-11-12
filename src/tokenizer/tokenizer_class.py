import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import xml.etree.ElementTree as ET
import re
from tqdm import tqdm
import pickle

def extract_wiki_text(xml_path, output_path):
    """
    Extract text from a wiki XML dump and write it to a file.

    The function takes a path to a wiki XML dump and a path to an output file.
    It extracts all the text from the XML dump, removes wiki markup, and writes it to the output file.

    Parameters
    ----------
    xml_path : str
        The path to the wiki XML dump
    output_path : str
        The path to the output file

    Returns
    -------
    None
    """
    patterns = [r"\{\{", r"\}\}", r"\[\[", r"\]\]", r"\*", r"\*\*", r"\=\=", r"\=\=\="]

    tree = ET.parse(xml_path)
    root = tree.getroot()
    with open(output_path, "w", encoding="utf-8") as f:
        for elem in root.iter():
            if elem.text and elem.text.strip():
                cleaned = elem.text.strip()
                for pattern in patterns:
                    cleaned = re.sub(pattern, "", cleaned)
                f.write(cleaned + "\n")

class BPETokenizer:
    default_vocab_size = 5000

    def __init__(self, vocab_size):
        """
        Initializes a BPETokenizer object.
        If input is None, vocab size is set to 1000.
            OUTDATED --> No longer uses input, instead, uses harcoded value of 32k vocab size
        """
        self.vocab_size = vocab_size or self.default_vocab_size # int(np.sqrt(len(input))) # default vocab_size set to 32000 for a large dataset
        self.base_vocab_size = 256
        self.merges = {} # initializing merges dictionary
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0,p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        self._ensure_vocab()
        
        self.eos_token_id = self.vocab_size
        self.vocab[self.eos_token_id] = "b<EOS>"
        self.vocab_size += 1

    def _rebuild_vocab(self):
        """
        Rebuilds the vocab dictionary based on the merges.

        This function should only be called internally when the merges dictionary is updated.
        It rebuilds the vocab dictionary by iterating over the merges in ascending order of idx.
        For each merge, it updates the vocab dictionary with the merged token.
        """
        vocab = {idx: bytes([idx]) for idx in range(self.base_vocab_size)}
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda item: item[1]):
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab

    def _ensure_vocab(self):
        """
        Ensures that the vocab dictionary is up-to-date with the merges dictionary.

        If any of the merged token ids are not in the vocab dictionary, this function rebuilds the vocab dictionary by calling _rebuild_vocab.

        This function should only be called internally when the merges dictionary is updated.
        """
        if any(idx not in self.vocab for idx in self.merges.values()):
            self._rebuild_vocab()

    def get_stats(self, input):
        """
        Given a text, returns a dictionary of pair counts.
        OPTIMIZED: Use defaultdict for faster counting.
        The key is a tuple of two adjacent characters, and the value is the count of that pair.
        """
        from collections import defaultdict
        counts = defaultdict(int)
        for pair in zip(input, input[1:]):
            counts[pair] += 1
        return dict(counts)

    def merge(self, input, pair, idx):
        """
        Merge a pair of adjacent ids in a list of ids to a single idx.
        OPTIMIZED: Use list comprehension for faster execution.

        Args:
            input (list): The list of ids to merge.
            pair (tuple): The pair of ids to merge.
            idx (int): The id to replace the pair with.

        Returns:
            list: The list of ids with the pair merged.
        """
        new_input = []
        i = 0
        pair_0, pair_1 = pair  # Unpack once to avoid repeated tuple indexing

        while i < len(input):
            # Check if we can merge at this position
            if i < len(input) - 1 and input[i] == pair_0 and input[i + 1] == pair_1:
                new_input.append(idx)
                i += 2
            else:
                new_input.append(input[i])
                i += 1

        return new_input

    def make_merges(self, input, dataset_length):
        """
        Merge adjacent ids in a list of ids until the vocab size is reached. Why? This is to increase the vocab size. This is to compress more tokens into a a single token, making the context length more compact, and the model can remember more at a time.
        Args:
            input (list): The list of ids to merge.
            dataset_length (int): The length of the dataset to consider for merges.

        Returns:
            list: The list of ids with adjacent ids merged until the vocab size is reached.
        """

        num_merges = self.vocab_size - self.base_vocab_size
        merges = {}
        input = list(input)
        base_vocab_start = self.base_vocab_size
        print("Starging merges now: ")
        for i in tqdm(range(num_merges)): # iterating over the number of merges  num_merges//100
            stats = self.get_stats(input[:dataset_length]) # getting stats
            # print("Got stats")
            if not stats:
                # print("Stats are empty, breaking")
                break # safety check
            pair = max(stats, key=stats.get) # getting the pair with the highest count (most repeated pair)
            # print("Most repeated pair: ", pair)
            idx = base_vocab_start + i
            # print(f"merging {pair} into a new token: {idx}")
            input = self.merge(input[:dataset_length], pair, idx) # merging the most repeated pair into one token
            merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.merges = merges
        self._ensure_vocab()
        return input

    def decode(self, ids):
        """
        Given a list of ids, returns the corresponding text.
        """
        self._ensure_vocab()
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8")
        return text

    def encode(self, text):
        """
        Given a string of text, returns the corresponding list of ids.
        OPTIMIZED: Precompute merge priorities and use efficient merging.
        """
        tokens = list(text.encode("utf-8"))

        # Early exit for short sequences
        if len(tokens) < 2:
            return tokens

        # Cache merge priorities (lower = merge earlier)
        # This avoids repeated dictionary lookups
        merge_priorities = {pair: idx for pair, idx in self.merges.items()}

        while len(tokens) >= 2:
            # Find all pairs and their positions in one pass
            pairs = []
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in merge_priorities:
                    pairs.append((merge_priorities[pair], pair, i))

            if not pairs:
                break

            # Get the pair with lowest merge index (highest priority)
            _, best_pair, _ = min(pairs)
            idx = merge_priorities[best_pair]

            # Merge in-place style (more efficient)
            tokens = self.merge(tokens, best_pair, idx)

        return tokens

def clean_alpaca_text(file_path):
    """
    Read Alpaca dataset and strip out 'Instruction:', 'Input:', and 'Output:' labels.
    Returns clean text with only the actual content.
    """
    clean_text = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Remove the labels
            if line.startswith("Instruction:"):
                line = line.replace("Instruction:", "").strip()
            elif line.startswith("Input:"):
                line = line.replace("Input:", "").strip()
            elif line.startswith("Output:"):
                line = line.replace("Output:", "").strip()

            # Keep the line if it has content
            if line:
                clean_text.append(line)

    return " ".join(clean_text)

def main():
    # extract_wiki_text('tokenizer_training_data/enwiki-latest-pages-articles-multistream1.xml-p1p41242', 'tokenizer_training_data/all_wiki_text.txt')
    # print("Extracted wiki text")

    # Choose your training data source:
    # Option 1: Wikipedia data
    # training_data = open("tokenizer_training_data/all_wiki_text.txt", "r").read()

    # Option 2: Alpaca data (cleaned)
    training_data = clean_alpaca_text("tokenizer_training_data/alpaca_sample_utf8.txt")
    print("Read and cleaned training data")

    tokens = training_data.encode("utf-8") # turns raw text (strings) into utf-8 encoded bytes stored inside tokens variable
    print("Encoded training data")
    # print(list(tokens)[:100])

    # Variable declaration (params for tokenizer class)
    dataset_length = len(tokens) # TODO: When ready, change dataset length to len(tokens) for final tokenizer training
    # TODO: When ready, change vocab size to 32000 for final tokenizer training
    vocab_size = 2500
    print("Set dataset length and vocab size")

    tokenizer = BPETokenizer(vocab_size) # instancing the tokenizer class with tokens as the training data
    print("Initialized tokenizer")

    ids = tokenizer.make_merges(tokens, dataset_length) # calls make_merges function from inside tokenizer class and passes tokens
    # print("Merged tokens")

    # printing stats about the tokens for comparison
    print("Original tokens length:", len(tokens[:dataset_length]))
    print("Final ids length:", len(ids))
    print(f"Compression ratio: {len(tokens[:dataset_length]) / len(ids):.2f}X")

    with open("artifacts/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f) # turning the tokenizer object into a pickle file

    # with open('artifacts/tokenizer.pkl', 'rb') as f:
    #     tokenizer = pickle.load(f) # loading the tokenizer object from the pickle file

def test_tokenizer(path):
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    training_texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            training_texts.append(line.strip())
    

    ids = []
    for i in training_texts:
        ids.append(tokenizer.encode(i))

    print(ids)

if __name__ == "__main__":
    main()
    # test_tokenizer("tokenizer_training_data/alpaca_sample_utf8.txt")
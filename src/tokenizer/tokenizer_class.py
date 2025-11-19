import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import xml.etree.ElementTree as ET
import json
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
        """
        self.target_vocab_size = vocab_size or self.default_vocab_size # int(np.sqrt(len(input))) # default vocab_size set to 32000 for a large dataset
        self.base_vocab_size = 256
        self.merges = {} # initializing merges dictionary
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self._vocab_dirty = False  # Track if vocab needs rebuilding

        self.eos_token_id = self.target_vocab_size - 1
        self.vocab[self.eos_token_id] = b'<EOS>'
        self.padding_token_id = 0

        self.vocab_size = self.target_vocab_size


    def _rebuild_vocab(self):
        """
        Rebuilds the vocab dictionary based on the merges.

        This function should only be called internally when the merges dictionary is updated.
        It rebuilds the vocab dictionary by iterating over the merges in ascending order of idx.
        For each merge, it updates the vocab dictionary with the merged token.
        """
        vocab = {idx: bytes([idx]) for idx in range(self.base_vocab_size)}
        
        # Process merges in order of their index to ensure dependencies are resolved
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda item: item[1]):
            # Check if both parent tokens exist in vocab
            if p0 not in vocab:
                # If p0 is a merged token that hasn't been processed yet, skip or handle
                print(f"Warning: Token {p0} not found in vocab when building token {idx}")
                # Try to build it recursively if it's a merge token
                if p0 >= self.base_vocab_size:
                    # Find the merge that creates p0
                    for (mp0, mp1), midx in self.merges.items():
                        if midx == p0:
                            if mp0 in vocab and mp1 in vocab:
                                vocab[p0] = vocab[mp0] + vocab[mp1]
                            break
            
            if p1 not in vocab:
                # If p1 is a merged token that hasn't been processed yet
                print(f"Warning: Token {p1} not found in vocab when building token {idx}")
                # Try to build it recursively if it's a merge token
                if p1 >= self.base_vocab_size:
                    # Find the merge that creates p1
                    for (mp0, mp1), midx in self.merges.items():
                        if midx == p1:
                            if mp0 in vocab and mp1 in vocab:
                                vocab[p1] = vocab[mp0] + vocab[mp1]
                            break
            
            # Now attempt to create the merged token if both parents exist
            if p0 in vocab and p1 in vocab:
                vocab[idx] = vocab[p0] + vocab[p1]
            else:
                print(f"Error: Cannot create token {idx} from tokens {p0} and {p1}")
        
        vocab[self.eos_token_id] = b"<EOS>"

        self.vocab = vocab
        self._vocab_dirty = False

    def _ensure_vocab(self):
        """
        Ensures that the vocab dictionary is up-to-date with the merges dictionary.
        Only rebuilds if the vocab is marked as dirty.

        This function should only be called internally when the merges dictionary is updated.
        """
        if self._vocab_dirty:
            self._rebuild_vocab()

    def get_stats(self, input):
        """
        Given a text, returns a dictionary of pair counts.
        OPTIMIZED: Use direct iteration without numpy overhead.
        The key is a tuple of two adjacent characters, and the value is the count of that pair.
        """
        from collections import Counter

        if len(input) < 2:
            return {}

        # Use Counter for fastest counting - it's implemented in C
        return dict(Counter(zip(input, input[1:])))

    def merge(self, input, pair, idx):
        """
        Merge a pair of adjacent ids in a list of ids to a single idx.
        OPTIMIZED: Fast single-pass merge with pre-allocated list.

        Args:
            input (list): The list of ids to merge.
            pair (tuple): The pair of ids to merge.
            idx (int): The id to replace the pair with.

        Returns:
            list: The list of ids with the pair merged.
        """
        if len(input) < 2:
            return input

        pair_0, pair_1 = pair
        new_input = []
        i = 0
        input_len = len(input)

        while i < input_len:
            # Check if we can merge at this position
            if i < input_len - 1 and input[i] == pair_0 and input[i + 1] == pair_1:
                new_input.append(idx)
                i += 2
            else:
                new_input.append(input[i])
                i += 1

        return new_input

    def make_merges(self, input, dataset_length, min_freq_threshold=2):
        """
        Merge adjacent ids in a list of ids until the vocab size is reached.
        OPTIMIZED: Incremental statistics tracking + heap for O(log n) max finding.

        Args:
            input (list): The list of ids to merge.
            dataset_length (int): The length of the dataset to consider for merges.
            min_freq_threshold (int): Minimum frequency for a pair to be merged (default: 2).

        Returns:
            list: The list of ids with adjacent ids merged until the vocab size is reached.
        """
        from collections import defaultdict
        import heapq

        num_merges = self.vocab_size - self.base_vocab_size - 1 # -1 For EOS token id
        merges = {}
        input = list(input[:dataset_length])
        base_vocab_start = self.base_vocab_size
        print("Starting merges now: ")

        # Build initial pair statistics ONCE
        print("Building initial statistics...")
        stats = defaultdict(int)
        for i in range(len(input) - 1):
            pair = (input[i], input[i + 1])
            stats[pair] += 1

        # Build max heap (negate counts for max heap using Python's min heap)
        # Heap stores: (-count, pair)
        heap = [(-count, pair) for pair, count in stats.items()]
        heapq.heapify(heap)

        for merge_num in tqdm(range(num_merges)):
            # Find the most frequent pair - rebuild from stats if heap is empty or stale
            if not heap or len(heap) < len(stats) // 10:  # Rebuild if heap is mostly stale
                heap = [(-count, pair) for pair, count in stats.items() if count > 0]
                heapq.heapify(heap)

            if not heap or not stats:
                print(f"\nNo more pairs to merge. Stopping at {merge_num} merges.")
                break

            # Find the most frequent pair using heap
            pair = None
            max_count = 0
            while heap:
                neg_count, p = heapq.heappop(heap)
                count = -neg_count

                # Verify this pair is still valid (count hasn't changed)
                if p in stats and stats[p] == count:
                    pair = p
                    max_count = count
                    break

            if pair is None:
                # All heap entries were stale, rebuild from stats
                if stats:
                    pair = max(stats, key=stats.get)
                    max_count = stats[pair]
                else:
                    print(f"\nNo more pairs to merge. Stopping at {merge_num} merges.")
                    break

            # Early stopping: if the most frequent pair occurs less than threshold times
            if max_count < min_freq_threshold:
                print(f"\nMost frequent pair only appears {max_count} times. Stopping early at {merge_num} merges.")
                break

            idx = base_vocab_start + merge_num

            # Perform the merge with incremental statistics update
            pair_0, pair_1 = pair
            new_input = []
            i = 0

            # Track what pairs are affected by this merge for incremental stats update
            pairs_to_decrement = []
            pairs_to_increment = []

            while i < len(input):
                # Check if we can merge at this position
                if i < len(input) - 1 and input[i] == pair_0 and input[i + 1] == pair_1:
                    # Found a match - merge it

                    # Record pairs that will be removed
                    if i > 0:
                        # The pair before this merge position changes
                        old_left_pair = (input[i - 1], input[i])
                        pairs_to_decrement.append(old_left_pair)
                        new_left_pair = (input[i - 1], idx)
                        pairs_to_increment.append(new_left_pair)

                    if i + 2 < len(input):
                        # The pair after this merge position changes
                        old_right_pair = (input[i + 1], input[i + 2])
                        pairs_to_decrement.append(old_right_pair)
                        new_right_pair = (idx, input[i + 2])
                        pairs_to_increment.append(new_right_pair)

                    # The merged pair itself is removed
                    pairs_to_decrement.append(pair)

                    new_input.append(idx)
                    i += 2
                else:
                    new_input.append(input[i])
                    i += 1

            # Update statistics incrementally
            for p in pairs_to_decrement:
                stats[p] -= 1
                if stats[p] <= 0:
                    del stats[p]

            for p in pairs_to_increment:
                stats[p] += 1
                # Add new/updated pairs to heap
                heapq.heappush(heap, (-stats[p], p))

            input = new_input
            merges[pair] = idx
            
            if idx != self.eos_token_id:
                if pair[0] in self.vocab and pair[1] in self.vocab:
                    self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        self.merges = merges
        self.vocab[self.eos_token_id] = b"<EOS>" # Make sure that EOS token stays
        self._vocab_dirty = False
        return input

    def decode(self, ids):
        """
        Given a list of ids, returns the corresponding text.
        OPTIMIZED: Removed redundant _ensure_vocab() call - vocab is already built.
        """
        tokens = b"".join(self.vocab.get(idx, b"?") for idx in ids)
        text = tokens.decode("utf-8", errors='replace')
        return text

    def encode(self, text):
        """
        Given a string of text, returns the corresponding list of ids.
        Heap-based approach for O(n log n) performance instead of O(n²).
        """
        import heapq

        tokens = list(text.encode("utf-8"))

        if len(tokens) < 2:
            return tokens
        
        heap = []
        pair_at_pos = {}  # Track which pair is at each position

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merges:
                merge_idx = self.merges[pair]
                heapq.heappush(heap, (merge_idx, i, pair))
                pair_at_pos[i] = pair

        # Process merges in priority order (lowest merge_idx first)
        while heap:
            merge_idx, pos, pair = heapq.heappop(heap)

            # Make sure that this position hasn't been invalidated by previous merges
            if pos >= len(tokens) - 1:
                continue
            if (tokens[pos], tokens[pos + 1]) != pair:
                continue

            # Perform the merge in-place
            tokens[pos] = merge_idx
            del tokens[pos + 1]

            # Update pair_at_pos tracking - remove invalidated entries
            if pos in pair_at_pos:
                del pair_at_pos[pos]
            if pos + 1 in pair_at_pos:
                del pair_at_pos[pos + 1]

            # Check and add new pair to the left
            if pos > 0 and pos - 1 not in pair_at_pos:
                new_pair = (tokens[pos - 1], tokens[pos])
                if new_pair in self.merges:
                    new_merge_idx = self.merges[new_pair]
                    heapq.heappush(heap, (new_merge_idx, pos - 1, new_pair))
                    pair_at_pos[pos - 1] = new_pair

            # Check and add new pair to the right
            if pos < len(tokens) - 1 and pos not in pair_at_pos:
                new_pair = (tokens[pos], tokens[pos + 1])
                if new_pair in self.merges:
                    new_merge_idx = self.merges[new_pair]
                    heapq.heappush(heap, (new_merge_idx, pos, new_pair))
                    pair_at_pos[pos] = new_pair

        return tokens
    
    def export_to_json(self, path):
        pass

def clean_text(file_path):
    """
    Read dataset and strip out labels.
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
            elif line.startswith("Response:"):
                line = line.replace("Response:", "").strip()
            elif line.startswith("Context:"):
                line = line.replace("Context:", "").strip()

            # Keep the line if it has content
            if line:
                clean_text.append(line)

    return " ".join(clean_text)

def main():
    # extract_wiki_text('tokenizer_training_data/enwiki-latest-pages-articles-multistream1.xml-p1p41242', 'tokenizer_training_data/all_wiki_text.txt')
    # print("Extracted wiki text")

    training_data = clean_text("training_data/alpaca.txt")
    print("Read and cleaned training data")

    tokens = training_data.encode("utf-8") # turns raw text (strings) into utf-8 encoded bytes stored inside tokens variable
    print("Encoded training data")
    # print(list(tokens)[:100])

    # Variable declaration (params for tokenizer class)
    dataset_length = len(tokens)
    vocab_size = 32000
    print("Set dataset length and vocab size")

    tokenizer = BPETokenizer(vocab_size) # instancing the tokenizer class with tokens as the training data
    print("Initialized tokenizer")

    ids = tokenizer.make_merges(tokens, dataset_length) # calls make_merges function from inside tokenizer class and passes tokens
    # print("Merged tokens")

    # printing stats about the tokens for comparison
    print("Original tokens length:", len(tokens[:dataset_length]))
    print("Final ids length:", len(ids))
    print(f"Compression ratio: {len(tokens[:dataset_length]) / len(ids):.2f}X")

    with open("artifacts/tokenizer/tokenizer_alpaca.pkl", "wb") as f:
        pickle.dump(tokenizer, f) # turning the tokenizer object into a pickle file

    # with open('artifacts/tokenizer.pkl', 'rb') as f:
    #     tokenizer = pickle.load(f) # loading the tokenizer object from the pickle file

def tokenize_training_data(path, batch_size=1000):
    """
    Tokenizes training data and saves it into a pickled file to save time in training.
    OPTIMIZED: Batch processing to reduce overhead and memory allocations.

    Args:
        path (str): Path to the training data
        batch_size (int): Number of lines to process in each batch (default: 1000)
    """
    with open("artifacts/tokenizer/tokenizer_dolly_15k.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    # Pre-compile label removal for faster processing
    labels = ["Instruction:", "Input:", "Output:", "Response:", "Context:"]

    tokenized_lines = []
    batch = []

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing training data..."):
            line = line.strip()
            if not line:
                continue

            # Remove labels
            for label in labels:
                line = line.replace(label, "")
            line = line.strip()

            if line:
                batch.append(line)

                # Process batch when it reaches batch_size
                if len(batch) >= batch_size:
                    for text in batch:
                        ids = tokenizer.encode(text)
                        tokenized_lines.append(ids)
                    batch = []

        # Process remaining lines in batch
        if batch:
            for text in batch:
                ids = tokenizer.encode(text)
                tokenized_lines.append(ids)

    # Flatten the list of lists into one continuous 1D array more efficiently
    # First calculate total length to pre-allocate array
    total_length = sum(len(line) for line in tokenized_lines)
    flat = np.empty(total_length, dtype=np.int32)

    # Fill the array in one pass
    idx = 0
    for line in tokenized_lines:
        line_len = len(line)
        flat[idx:idx + line_len] = line
        idx += line_len

    # Save as memmap
    fp = np.memmap("artifacts/tokenizer/tokenized_dolly_15k.dat", dtype="int32", mode="w+", shape=flat.shape)
    fp[:] = flat[:]
    del fp

    print(f"Saved {flat.shape[0]} tokens as flat 1D memmap.")



def test_tokenizer(path):
    with open("artifacts/tokenizer/tokenizer_general_knowledge.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    # with open(path, "rb") as f:
    #     tokenized_training_data = pickle.load(f)

    print(f"Token 0: {tokenizer.decode([0])}")

    ids = tokenizer.encode("Hello World")
    print(ids)
    text = tokenizer.decode(ids)
    print(text)

    # for token_ids in tokenized_training_data[:10]:
    #     decoded_text = tokenizer.decode(token_ids)
    #     print("tokens: ", token_ids)
    #     print("text: ", decoded_text)

if __name__ == "__main__":
    main()
    # tokenize_training_data("training_data/pygpt_training_corpus.txt")
    # test_tokenizer("artifacts/tokenized_training_data.pkl")
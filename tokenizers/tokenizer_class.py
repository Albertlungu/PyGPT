import numpy as np
import xml.etree.ElementTree as ET
import re
from tqdm import tqdm

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
    patterns = [r"\{\{", r"\}\}", r"\[\[", r"\]\]", r"\*", r"\*\*", r"\=\="]

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
    def __init__(self, input=32000):
        """
        Initializes a BPETokenizer object.
        If input is None, vocab size is set to 1000.
            OUTDATED --> No longer uses input, instead, uses harcoded value of 32k vocab size
        """
        self.vocab_size = 32000 # int(np.sqrt(len(input))) # default vocab_size set to 32000 for a large dataset
        self.base_vocab_size = 256
        self.merges = {} # initializing merges dictionary

    def get_stats(self, input):
        """
        Given a text, returns a dictionary of pair counts.
        The key is a tuple of two adjacent characters, and the value is the count of that pair.
        """
        counts = {} # initializing counts dictionary
        for pair in zip(input, input[1:]): # zipping characters that are one next to another (imagine a zipper, how the teeth thread --> this is what zip() does) 
            counts[pair] = counts.get(pair, 0) + 1 # counts the amount of adjacent pairs in a text
        return counts

    def merge(self, input, pair, idx):
        """
        Merge a pair of adjacent ids in a list of ids to a single idx.

        Args:
            input (list): The list of ids to merge.
            pair (tuple): The pair of ids to merge.
            idx (int): The id to replace the pair with.

        Returns:
            list: The list of ids with the pair merged.
        """
        new_input = [] # input after merging adjacent tokens
        i = 0
        while i < len(input):
            if i < len(input) - 1 and input[i] == pair[0] and input[i + 1] == pair[1]: # checks if the current index is equal to the first element of the pair and the next index is equal to the second element of the pair
                new_input.append(idx)
                i += 2
            else:
                new_input.append(input[i])
                i += 1
        return new_input
    
    def make_merges(self, input):
        """
        Merge adjacent ids in a list of ids until the vocab size is reached. Why? This is to increase the vocab size. This is to compress more tokens into a a single token, making the context length more compact, and the model can remember more at a time.
        Args:
            input (list): The list of ids to merge.

        Returns:
            list: The list of ids with adjacent ids merged until the vocab size is reached.
        """
        
        num_merges = self.vocab_size - self.base_vocab_size
        merges = {}
        input = list(input)
        base_vocab_start = self.base_vocab_size
        for i in tqdm(range(num_merges)): # iterating over the number of merges using tqdm in order to show progress bar
            stats = self.get_stats(input) # getting stats
            if not stats:
                break # safety check
            pair = max(stats, key=stats.get) # getting the pair with the highest count (most repeated pair)
            idx = base_vocab_start + i
            # print(f"merging {pair} into a new token: {idx}")  
            input = self.merge(input, pair, idx) # merging the most repeated pair into one token
            merges[pair] = idx
        self.merges = merges
        return input

if __name__ == "__main__":
    training_data = open("tokenizer_training_data/all_wiki_text.txt", "r").read() # reading training data from wiki file

    tokens = training_data.encode("utf-8") # turns raw text (strings) into utf-8 encoded bytes stored inside tokens variable

    tokenizer = BPETokenizer(tokens) # isntancing the tokenizer class with tokens as the training data

    ids = tokenizer.make_merges(tokens) # calls make_merges function from inside tokenizer class and passes tokens


    # printing stats about the tokens for comparison
    print("Original tokens length:", len(tokens))
    print("Final ids length:", len(ids))
    print(f"Compression ratio: {len(tokens) / len(ids):.2f}X")
    # pass
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from user_input import UserInput
import xml.etree.ElementTree as ET
from tokenizers.tokenizer_class import BPETokenizer
import re
from tqdm import tqdm
import pickle

"""
What to know:    
- The embedding model is a matrix which takes in (vocab_size, embedding size) as parameters for the size of the matrix
    - Embedding size tends to be an exponent of 2 (i.e. 128, 256, 512, 1024, etc.)
Very useful website for understanding this:
https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
"""

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim, max_seq_length = 512):
        """
        Initializes an EmbeddingLayer object.

        Parameters:
            vocab_size (int): the size of the vocabulary
            embedding_dim (int): the size of the embedding dimension
            max_seq_length (int, optional): the maximum sequence length. Defaults to 512.
        """
        self.vocab_size = vocab_size # taken from the tokenizer.pkl inside of artifacts/tokenizer.pkl or from inside tokenizers/tokenizer_class.py
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

        self.embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * np.sqrt(1.0/self.vocab_size) # Basically, random numbers are selected for the vectors right now as placeholder so that the algorithm doesn't see symmetry and simply assign the same vector values to every word upon training


def main():

    with open('artifacts/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()
    
    print(tokenizer.vocab_size)

    # Initialize the embedding layer
    embedding_layer = EmbeddingLayer(vocab_size=tokenizer.vocab_size, embedding_dim = 256)

    # printing the embeddings to understand wtf is going on
    print(embedding_layer.embeddings)

if __name__ == '__main__':
    main()
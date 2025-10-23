import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from user_input import UserInput
import xml.etree.ElementTree as ET
from tokenizers.tokenizer_class import BPETokenizer
import re
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

"""
What to know:    
- The embedding model is a matrix which takes in (vocab_size, embedding size) as parameters for the size of the matrix
    - Embedding size tends to be an exponent of 2 (i.e. 128, 256, 512, 1024, etc.)
Very useful website for understanding this:
https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
"""


class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim, max_seq_length=512, n=10000):
        """
        Initializes an EmbeddingLayer object.

        Parameters:
            vocab_size (int): the size of the vocabulary
            embedding_dim (int): the size of the embedding dimension
            max_seq_length (int, optional): the maximum sequence length. Defaults to 512.
        """
        self.vocab_size = vocab_size  # taken from the tokenizer.pkl inside of artifacts/tokenizer.pkl or from inside tokenizers/tokenizer_class.py
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.n = n

        self.embeddings = np.random.randn(
            self.vocab_size, self.embedding_dim
        ) * np.sqrt(
            1.0 / self.vocab_size
        )  # Basically, random numbers are selected for the vectors right now as placeholder so that the algorithm doesn't see symmetry and simply assign the same vector values to every word upon training

        self.positional_encodings = self._create_positional_encoding(
            n
        )  # using the function that will be declared later to get the positional encoding of a certain word

        self.last_input_ids = (
            None  # for future backpropagation, storing the last input ID
        )

        self.encoding_gradient = np.zeros_like(
            self.embeddings
        )  # for future backpropagation, storing the gradient of the encoding

    def _create_positional_encoding(self, n=10000):
        # making variables in accordance to understanding_positional_encoding.md
        """
        Creates positional encoding for the given embedding layer.

        Parameters:
            n (int, optional): the maximum sequence length. Defaults to 10000.

        Returns:
            P (numpy.array): the positional encoding function
        """
        L = self.max_seq_length  # length of the embeddings inside the embedding layer
        d = (
            self.embedding_dim
        )  # dimension of the embedding (amount of # in the vectors)
        P = np.zeros((L, d))  # positional encoding function
        for k in range(L):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        return P


def main():
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    n = 10000

    # Initialize the embedding layer
    embedding_layer = EmbeddingLayer(
        vocab_size=tokenizer.vocab_size, embedding_dim=256, n=n
    )

    print("vocab size: ", tokenizer.vocab_size)

    # printing the embeddings to understand wtf is going on
    print(embedding_layer.embeddings[:5])
    print("positional encodings: ", embedding_layer.positional_encodings[:5])
    print(embedding_layer.last_input_ids)
    print(embedding_layer.encoding_gradient)

    P = embedding_layer.positional_encodings
    cax = plt.matshow(P)
    plt.gcf().colorbar(cax)
    plt.show()


if __name__ == "__main__":
    main()

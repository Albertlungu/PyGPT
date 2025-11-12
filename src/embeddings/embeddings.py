import os
import numpy as np
import jax
import jax.numpy as jnp

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import xml.etree.ElementTree as ET
from src.tokenizer.tokenizer_class import BPETokenizer
from src.embeddings.positional_encoding import PositionalEncoding
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

    default_embedding_dim = 512

    def __init__(self, vocab_size = None, embedding_dim = None, max_seq_length = 512, n = 10000):
        """
        Initializes an EmbeddingLayer object.

        Parameters:
            vocab_size (int): the size of the vocabulary. Optional.
            embedding_dim (int): the size of the embedding dimension. Optional.
            max_seq_length (int, optional): the maximum sequence length. Defaults to 512. Optional.
        """
        self.vocab_size = vocab_size or BPETokenizer.default_vocab_size# taken from the tokenizer.pkl inside of artifacts/tokenizer.pkl or from inside tokenizers/tokenizer_class.py
        self.embedding_dim = embedding_dim or self.default_embedding_dim
        self.max_seq_length = max_seq_length
        self.n = n

        # Create key inside __init__, not at class level
        self.key = jax.random.PRNGKey(0)
        self.embeddings = jax.random.normal(self.key, (self.vocab_size, self.embedding_dim)) * jnp.sqrt(1.0/self.vocab_size) # Basically, random numbers are selected for the vectors right now as placeholder so that the algorithm doesn't see symmetry and simply assign the same vector values to every word upon training

        self.positional_encoding_class = PositionalEncoding(self.embedding_dim, self.max_seq_length)
        self.positional_encodings = self.positional_encoding_class._create_positional_encoding(n) # using the function that will be declared later to get the positional encoding of a certain word

    @staticmethod
    def pad_token_ids(max_len, token_ids, pad_token_id=0):
        batch_size = len(token_ids)
        padded = jnp.full((batch_size, max_len), pad_token_id, dtype=jnp.float32)
        lengths = jnp.array([min(len(seq), max_len) for seq in token_ids])

        def scatter_seq(i,seq):
            l = lengths[i]
            return padded.at[i, :l].set(jnp.array(seq[:l]))

        padded = jax.lax.fori_loop(0, batch_size, scatter_seq, padded)
        return padded
        

    @staticmethod
    @jax.jit
    def embedding_fwd(params, padded_token_ids, pad_token_id=0):
        embeddings, positional_encodings = params # params[0] = embeddings; params[1] = positional_encodings, is a tuple of the two.

        mask = padded_token_ids != pad_token_id
        token_embeddings = embeddings[padded_token_ids]
        token_embeddings = token_embeddings * jnp.sqrt(embeddings.shape[1])

        pos_enc = positional_encodings[:padded_token_ids.shape[1], :]
        output = token_embeddings + pos_enc[None, :, :]
        return output, mask
    
    # Removed loss_fn method - loss computation is now handled in Trainer

    
    def update(self, grads, learning_rate):
        """
        Updates embedding weights using gradients (added up from all ids)

        Args:
            learning_rate (float): rate at which the machine moves forward
        """
        embedding_grads, pos_enc_grads = grads
        self.embeddings -= learning_rate * embedding_grads
        self.positional_encodings -= learning_rate * pos_enc_grads
    
    def save(self, filepath):
        """
        Save embeddings to a file
        """

        with open(filepath, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'max_seq_length': self.max_seq_length
            }, f)

    def load(self, filepath):
        """Load embeddings from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.vocab_size = data['vocab_size']
            self.embedding_dim = data['embedding_dim']
            self.max_seq_length = data['max_seq_length']
            self.positional_encodings = self.positional_encoding_class._create_positional_encoding()

    def get_params_and_grads(self):
        return (self.embeddings, self.positional_encodings)


def main():
    # print("="*60)
    # print("LOADING TOKENIZER")
    # print("="*60)
    
    with open('artifacts/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()
    
    print("="*60)
    print("Main ran with no errors")
    print("="*60)

if __name__ == '__main__':
    main()
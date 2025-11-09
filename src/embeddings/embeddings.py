import numpy as np
import sys
import os
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

    default_embedding_dim = 256
    
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

        self.embeddings = (np.random.randn(self.vocab_size, self.embedding_dim) * np.sqrt(1.0/self.vocab_size)).astype(np.float32) # Basically, random numbers are selected for the vectors right now as placeholder so that the algorithm doesn't see symmetry and simply assign the same vector values to every word upon training

        self.positional_encoding_class = PositionalEncoding(self.embedding_dim, self.max_seq_length)

        self.positional_encodings = self.positional_encoding_class._create_positional_encoding(n) # using the function that will be declared later to get the positional encoding of a certain word

        self.last_input_ids = None # for future backpropagation, storing the last input ID

        self.encoding_gradient = np.zeros_like(self.embeddings) # for future backpropagation, storing the gradient of the encoding

    def fwd(self, token_ids):
        """
        Forward pass to convert input_ids to embeddings

        Args:
            token_ids (list or array): list or array of sequences of token IDs (can be 1D or 2D with variable lengths)
        
        Return:
            output (array): embeddings that have positional encoding information inside of them (gradient)
        """
        # Ensure token_ids is a list of sequences
        if isinstance(token_ids, np.ndarray) and token_ids.ndim == 2:
            sequences = [list(seq) for seq in token_ids]
        elif isinstance(token_ids, (list, tuple)):
            # Check if it's a list of ints or list of lists
            if len(token_ids) == 0:
                sequences = []
            elif isinstance(token_ids[0], (list, tuple, np.ndarray)):
                sequences = [list(seq) for seq in token_ids]
            else:
                sequences = [list(token_ids)]
        else:
            sequences = [list(token_ids)]

        self.batch_size = len(sequences)
        self.max_len = max(len(seq) for seq in sequences) if sequences else 0
        self.max_len = min(self.max_len, self.max_seq_length)  # limit to max_seq_length

        # Initialize padded array with zeros (assumed padding token id = 0)
        padded_token_ids = np.zeros((self.batch_size, self.max_len), dtype=int)

        for i, seq in enumerate(sequences):
            length = min(len(seq), self.max_len)
            padded_token_ids[i, :length] = seq[:length]

        self.last_input_ids = padded_token_ids

        token_embeddings = self.embeddings[padded_token_ids]  # shape (batch_size, max_len, embedding_dim)
        token_embeddings = token_embeddings * np.sqrt(self.embedding_dim)  # scale embeddings

        pos_enc = self.positional_encodings[:self.max_len, :]  # shape (max_len, embedding_dim)

        # Add positional encoding to each sequence
        output = token_embeddings + pos_enc[np.newaxis, :, :]

        return output  # shape (batch_size, max_len, embedding_dim)

    def backward(self, gradient):
        """
        The backward pass to convert embeddings back to ids

        Args:
            gradient (array): the embeddings with positional encodings to convert
        """
        if gradient.ndim == 2:
            gradient = gradient[np.newaxis, :]

        gradient = gradient * np.sqrt(self.embedding_dim) # scaling it down in backward pass because we did this in forward pass
        batch_size, seq_len, _ = gradient.shape
        
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = self.last_input_ids[b,s]
                self.encoding_gradient[token_id] += gradient[b,s]

        return self.encoding_gradient
    
    def update(self, learning_rate):
        """
        Updates embedding weights using gradients (added up from all ids)

        Args:
            learning_rate (float): rate at which the machine moves forward
        """
        self.embeddings -= learning_rate*self.encoding_gradient
        self.encoding_gradient.fill(0)
    
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
            self.positional_encodings = self._create_positional_encoding()

    def get_params_and_grads(self):
        return [{'value': self.embeddings, 'grad': self.encoding_gradient}]


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
    
    # n = 10000
    # embedding_dim = 256
    # embedding_layer = EmbeddingLayer(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim, n=n)
    
    # # filepath = "artifacts/embeddings.pkl"
    # # embedding_layer.save(filepath)
    # # print("saved encoding model to path: ", filepath)


    # sample_text = "Hello world"
    # for i in tokenizer.encode(sample_text):
    #     print(tokenizer.decode([i]))
    
    # embeddings = embedding_layer.embeddings[:2, :10]
    # print("What the embeddings look like (definition and meaning of word as a vector): ", embeddings)
    # pos_enc = embedding_layer.positional_encodings[:2, :10]
    # print("What positional encodings look like (vector describing position of word in sentence): ", pos_enc)
    # fwd = embedding_layer.forward(tokenizer.encode(sample_text))
    # print("What the forward function returns - a numpy array of embeddings/definitions of words: ", fwd[:2, :10])
    
    # print("What the backward function of a forward pass looks like: ", embedding_layer.backward(fwd)[:2])


if __name__ == '__main__':
    main()
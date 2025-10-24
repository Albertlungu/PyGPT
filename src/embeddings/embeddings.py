import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    def __init__(self, vocab_size, embedding_dim, max_seq_length = 512, n = 10000):
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
        self.n = n

        self.embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * np.sqrt(1.0/self.vocab_size) # Basically, random numbers are selected for the vectors right now as placeholder so that the algorithm doesn't see symmetry and simply assign the same vector values to every word upon training

        self.positional_encodings = self._create_positional_encoding(n) # using the function that will be declared later to get the positional encoding of a certain word

        self.last_input_ids = None # for future backpropagation, storing the last input ID

        self.encoding_gradient = np.zeros_like(self.embeddings) # for future backpropagation, storing the gradient of the encoding

    def _create_positional_encoding(self, n = 10000):
        # making variables in accordance to understanding_positional_encoding.md
        """
        Creates positional encoding for the given embedding layer.

        Parameters:
            n (int, optional): the maximum sequence length. Defaults to 10000.

        Returns:
            P (numpy.array): the positional encoding function, that describes the position of word in a given input
        """
        # L = self.max_seq_length # length of the embeddings inside the embedding layer
        # d = self.embedding_dim # dimension of the embedding (amount of # in the vectors)
        # P = np.zeros((L, d)) # positional encoding function
        # for k in range(L):
        #     for i in np.arange(int(d/2)):
        #         denominator = np.power(n, 2*i/d)
        #         P[k, 2*i] = np.sin(k/denominator)
        #         P[k, 2*i +1] = np.cos(k/denominator)
        # return P
        
        L, d = self.max_seq_length, self.embedding_dim
        pos = np.arrange(L)[:, np.newaxis]
        i = np.arrange(d)[np.newaxis, :]
        angle_rates = 1 / np.power(n, (2 * (i//2)) / d)
        P = pos * angle_rates
        P[:, 0::2] = np.sin(P[:, 0::2])
        P[:, 1::2] = np.cos(P[:, 1::2])
        return P

    def forward(self, token_ids):
        """
        Forward pass to convert input_ids to embeddings

        Args:
            token_ids (array): array containing input IDs of query
        
        Return:
            output (array): embeddings that have positional encoding information inside of them (gradient)
        """
        token_ids = np.array((token_ids))
        if token_ids.ndim == 1: # checks if token_ids array is 1 or 2 dimensions (1D vs 2D array => [x,y,z] v.s. [[x,y],[a,b]])
            token_ids = token_ids[np.newaxis, :]
            squeeze_dim = True # To remember to remove the second dimension from the token_ids at the end
        else:
            squeeze_dim = False # no need to remove second dimension

        batch_size, seq_len = token_ids.shape # gets dimensions of input, e.g. if token_ids shape is (2,10), batch_size=2 and seq_length=10
        self.last_input_ids = token_ids # saves this for backward pass later

        token_embeddings = self.embeddings[token_ids] # replaces each ID with its corresponding vector from inside embeddings
        token_embeddings = token_embeddings * np.sqrt(self.embedding_dim) # balancing the size of positional encodings with embeddings

        pos_enc = self.positional_encodings[:seq_len, :] # slicing positional encodings to match the actual sequence length

        gradient = token_embeddings + pos_enc[np.newaxis, :, :] # ig "intertwining" positional encodings wth token embeddings
        
        if squeeze_dim:
            gradient = gradient.squeeze(0)
        
        return gradient

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

        return self.encoding_gradient[token_id]
    
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


def main():
    print("="*60)
    print("LOADING TOKENIZER")
    print("="*60)
    
    with open('artifacts/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()
    
    n = 10000
    embedding_dim = 256
    embedding_layer = EmbeddingLayer(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim, n=n)
    
    # filepath = "artifacts/embeddings.pkl"
    # embedding_layer.save(filepath)
    # print("saved encoding model to path: ", filepath)


    sample_text = "Hello world"
    for i in tokenizer.encode(sample_text):
        print(tokenizer.decode([i]))
    
    embeddings = embedding_layer.embeddings[:2, :10]
    print("What the embeddings look like (definition and meaning of word as a vector): ", embeddings)
    pos_enc = embedding_layer.positional_encodings[:2, :10]
    print("What positional encodings look like (vector describing position of word in sentence): ", pos_enc)
    fwd = embedding_layer.forward(tokenizer.encode(sample_text))
    print("What the forward function returns - a numpy array of embeddings/definitions of words: ", fwd[:2, :10])
    
    print("What the backward function of a forward pass looks like: ", embedding_layer.backward(fwd)[:2])


if __name__ == '__main__':
    main()
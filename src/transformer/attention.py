import numpy as np
import pickle
import sys
import os

# Add project root to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.embeddings.embeddings import EmbeddingLayer
from feed_forward import FeedForward

class Attention():
    def __init__(self, embedding_layer: EmbeddingLayer, ffn = FeedForward):
        self.embedding_dim = embedding_layer.embedding_dim
        self.batch_size = embedding_layer.batch_size
        self.seq_len = embedding_layer.max_len


        self.W_Q = np.randn(self.embedding_dim, self.embedding_dim)
        self.W_K = np.randn(self.embedding_dim, self.embedding_dim)
        self.W_V = np.randn(self.embedding_dim, self.embedding_dim)

        self.W_O = np.randn(self.embedding_dim, self.embedding_dim)

    def softmax(self, x):
        """Softmax function, returning normalized probabilities?'/

        Args:
            x (numpy array):  scaled scores from mask
        
        Return:
            probablities (numpy array): probabilities
        """

        e_x = np.exp(x)
        probabilities = e_x / np.sum(e_x, axis=-1, keepdims=True)

        return probabilities
    
    
    
def main():
    # Test the attention class
    embedding_layer = EmbeddingLayer()
    attention = Attention(embedding_layer)
    # print(f"Embedding dimension: {attention.embedding_dim}")

if __name__ == "__main__":
    main()
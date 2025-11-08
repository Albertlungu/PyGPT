import numpy as np
import pickle
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.embeddings.embeddings import EmbeddingLayer
from src.transformer.feed_forward import FeedForward
from src.tokenizer.tokenizer_class import BPETokenizer  # manually register the class
import matplotlib.pyplot as plt


class Attention():
    """
    Implements a single-head self-attention mechanism for transformer models.

    This class computes attention by generating query, key, and value matrices
    from input embeddings. It applies scaled dot-product attention with a 
    causal mask to prevent positions from attending to future tokens, 
    ensuring proper autoregressive behaviour in decoder models. 

    Attributes:
        input (3D tensor): Embedded input tokens of shape (batch_size, seq_len, embedding_dim).
        batch_size (int): Number of sequences in a batch.
        seq_len (int): Length of each input sequence.
        embedding_dim (int): Dimensionality of input embeddings.
        W_Q, W_K, W_V, W_O (3D tensor): Learnable weight matrices for queries, keys, values, and output projection.
        Q, K, V (3D tensor): Query, key, and value matrices computed from input embeddings.
        attention_scores (3D tensor): Raw attention scores before scaling or masking.
        scaled_scores (3D tensor): Attention scores scaled by sqrt(embedding_dim).
        masked_scores (3D tensor): Scaled scores with causal mask applied.
        attention_weights (3D tensor): Softmax-normalized attention weights.
        attention_output (3D tensor): Weighted sum of values based on attention weights.
        output (3D tensor): Final projected output of the attention layer.

    """
    def __init__(self, token_ids, embedding_layer: EmbeddingLayer):
        """
        Initializes Attention instance attributes

        Args:
            token_ids (list): IDs of input tokens given to model (padding is already applied in FeedForward layer)
            embedding_layer (EmbeddingLayer): EmbeddingLayer class. Takes no arguments.
        """
        self.embedding_dim = embedding_layer.embedding_dim
        embedded = embedding_layer.forward(token_ids)
        self.input = embedded
        self.batch_size = embedded.shape[0]  # batch dimension
        self.seq_len = embedded.shape[1]     

        self.W_Q = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01
        self.W_K = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01
        self.W_V = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01

        self.W_O = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01

    @staticmethod
    def softmax(x):
        """Softmax function, returning normalized probabilities

        Args:
            x (numpy array):  scaled scores from mask
        
        Return:
            probablities (numpy array): probabilities
        """

        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        probabilities = e_x / np.sum(e_x, axis=-1, keepdims=True)

        return probabilities

    def fwd(self, x = None):
        """
        Computing query, key, and value matrices from input embeddings.
        Using these, compute attention scores and scaled attention scores.
        Make mask (casual decoder).
            - The mask blocks attention to certain positions in the sequence
                - To prevent the model from looking ahead to predicted tokens
                - When predicting token *t*, the model can only look at tokens ≤ t

        Return:
            ouput (3D list/tensor): Represents each token's embedding after looking at all tokens in the sequence. Shape: (batch_size, seq_len, embedding_dim)
        """
        if x is not None:
            self.input = x

        self.Q = self.input @ self.W_Q
        self.K = self.input @ self.W_K
        self.V = self.input @ self.W_V

        # Creating attention scores based on tranposed key matrix shapes from (batch, seq_len, embedding_dim) to (batch, embedding_dim, seq_len)
        self.K_transposed = np.transpose(self.K, (0,2,1))
        self.attention_scores = self.Q @ self.K_transposed # Shape result of this is (batch, seq_len, seq_len)

        # Scaling the scores to prevent dot product from becoming too large
        self.scaled_scores = self.attention_scores / np.sqrt(self.embedding_dim)

        # Applying a mask of shape (batch, seq_len, seq_len)
        mask = np.tril(np.ones((self.batch_size, self.seq_len, self.seq_len)))
        self.masked_scores = np.where(mask == 0, -1e9, self.scaled_scores)

        # Applying softmax function to masked scores
        self.attention_weights = self.softmax(self.masked_scores)

        # Applying attention to values
        self.attention_output = self.attention_weights @ self.V

        # Final output projection
        self.output = self.attention_output @ self.W_O
        return self.output

    
def main():
    # Test the attention class
    embedding_layer = EmbeddingLayer()
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()
    
    sample_texts = [
        "Hello World. My name is Albert Lungu",
        "What is your name?",
        "I like LLMs"
    ]

    # Convert texts to token ids
    token_ids = [tokenizer.encode(text) for text in sample_texts]

    attention = Attention(token_ids, embedding_layer)
    print(attention.fwd())

    attention_weights = attention.attention_weights[0]

    # plt.figure(figsize=(8, 6))
    # plt.imshow(attention_weights, cmap='viridis', interpolation='nearest')
    # plt.colorbar(label='Attention weight')
    # plt.title("Attention Heatmap (first sequence)")
    # plt.xlabel("Key position")
    # plt.ylabel("Query position")
    # plt.show()


if __name__ == "__main__":
    main()
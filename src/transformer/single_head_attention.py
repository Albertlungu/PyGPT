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
        embedded = embedding_layer.fwd(token_ids)
        self.input = embedded
        self.batch_size = embedded.shape[0]  # batch dimension
        self.seq_len = embedded.shape[1]     

        self.W_Q = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01
        self.W_K = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01
        self.W_V = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01
        self.W_O = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01

        self.dW_Q = np.zeros_like(self.W_Q)
        self.dW_K = np.zeros_like(self.W_K)
        self.dW_V = np.zeros_like(self.W_V)
        self.dW_O = np.zeros_like(self.W_O)

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
        return self.output, self.input, self.attention_output, self.V, self.attention_weights, self.K
    
    def backward(self, d_output, input_tensor, attention_output_fwd, V_fwd, attention_weights_fwd, K_fwd):
        """
        Backward pass for single-head attention.
        
        Args:
            d_output (numpy array): Gradient of the loss with respect to the attention output.
                                    Shape: (batch_size, seq_len, embedding_dim)
            input_tensor (numpy array): The input to the forward pass of the attention layer.
                                        Shape: (batch_size, seq_len, embedding_dim)
            attention_output_fwd (numpy array): The output of the attention mechanism from the forward pass.
                                                Shape: (batch_size, seq_len, embedding_dim)
            V_fwd (numpy array): The value matrix from the forward pass.
                                 Shape: (batch_size, seq_len, embedding_dim)
            attention_weights_fwd (numpy array): The attention weights from the forward pass.
                                                 Shape: (batch_size, seq_len, seq_len)
            K_fwd (numpy array): The key matrix from the forward pass.
                                 Shape: (batch_size, seq_len, embedding_dim)
        
        Returns:
            dX (numpy array): Gradient with respect to input embeddings. Same shape as self.input
        """
        # Gradient through output projection
        # output = attention_output @ W_O
        self.dW_O = np.sum(attention_output_fwd.transpose(0, 2, 1) @ d_output, axis=0)
        d_attention_output = d_output @ self.W_O.T

        # Gradient through attention weights multiplication
        # attention_output = attention_weights @ V
        d_attention_weights = d_attention_output @ V_fwd.transpose(0, 2, 1)
        dV = attention_weights_fwd.transpose(0, 2, 1) @ d_attention_output

        # Backprop through softmax (vectorized version for performance)
        # attention_weights = softmax(masked_scores)
        # Vectorized softmax gradient: dL/dx = p * (dL/dp - sum(p * dL/dp))
        # This is mathematically equivalent to the Jacobian version but much faster
        d_masked_scores = attention_weights_fwd * (
            d_attention_weights - np.sum(d_attention_weights * attention_weights_fwd, axis=-1, keepdims=True)
        )
        batch_size = attention_weights_fwd.shape[0]
        seq_len = attention_weights_fwd.shape[1]

        # Apply mask: masked positions do not backprop
        mask = np.tril(np.ones((seq_len, seq_len)))
        d_scaled_scores = d_masked_scores * mask

        # Gradient through scaling
        d_attention_scores = d_scaled_scores / np.sqrt(self.embedding_dim)

        # Gradient through Q @ K^T
        # Compute Q from input_tensor to match batch size
        Q = input_tensor @ self.W_Q
        dQ = d_attention_scores @ K_fwd
        dK = d_attention_scores.transpose(0, 2, 1) @ Q

        # Gradients w.r.t. weights
        self.dW_Q = np.sum(input_tensor.transpose(0, 2, 1) @ dQ, axis=0)
        self.dW_K = np.sum(input_tensor.transpose(0, 2, 1) @ dK, axis=0)
        self.dW_V = np.sum(input_tensor.transpose(0, 2, 1) @ dV, axis=0)

        # Gradient w.r.t input embeddings
        dX = dQ @ self.W_Q.T + dK @ self.W_K.T + dV @ self.W_V.T

        return dX
    
    def get_params_and_grads(self):
        return [
            {'value': self.W_Q, 'grad': self.dW_Q},
            {'value': self.W_K, 'grad': self.dW_K},
            {'value': self.W_V, 'grad': self.dW_V},
            {'value': self.W_O, 'grad': self.dW_O},
        ]

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
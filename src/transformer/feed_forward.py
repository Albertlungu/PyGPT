import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.embeddings.embeddings import EmbeddingLayer
from src.tokenizer.tokenizer_class import BPETokenizer 

class FeedForward():
    """
    A FeedForward neural network module used within transformer architectures.

    This class implements a two-layer feed-forward network with GELU activation.
    It takes token embeddings as input and applies a linear transformation followed
    by a non-linear activation and another linear transformation to produce the output.

    Key components:
    - Two sets of weights and biases (W1, B1 for the first layer, W2, B2 for the second layer).
    - GELU activation function.
    - Forward and backward passes for training with gradient descent.
    """

    def __init__(self, token_ids, embeddings: EmbeddingLayer):
        """
        Initializes the FeedForward network.

        Args:
            token_ids (list): List of token IDs representing input sequences.
            embeddings (EmbeddingLayer): An instance of EmbeddingLayer to convert token IDs to embeddings.

        Attributes:
            embedding_dim (int): Dimensionality of the embeddings.
            ff_dim (int): Dimensionality of the feed-forward hidden layer (4 times embedding_dim).
            W1 (np.ndarray): Weight matrix for the first linear layer of shape (embedding_dim, ff_dim).
            B1 (np.ndarray): Bias vector for the first linear layer of shape (ff_dim,).
            W2 (np.ndarray): Weight matrix for the second linear layer of shape (ff_dim, embedding_dim).
            B2 (np.ndarray): Bias vector for the second linear layer of shape (embedding_dim,).
            ff_input (np.ndarray): Input embeddings to the feed-forward network.
            hidden_layer (np.ndarray): Output of the first linear layer before activation.
            activated_layer (np.ndarray): Output after applying the GELU activation.
            output (np.ndarray): Final output of the feed-forward network.
        """
        self.embedding_dim = EmbeddingLayer.default_embedding_dim
        self.ff_dim = self.embedding_dim * 4 # Feed Forward dimension

        # Layers
        self.W1 = (np.random.randn(self.embedding_dim, self.ff_dim) * (1/np.sqrt(self.embedding_dim))).astype(np.float32) # Weight first layer of shape (embedding_dim, ff_dim)
        self.B1 = np.zeros(self.ff_dim, dtype=np.float32)# Bias first layer
        self.W2 = (np.random.randn(self.ff_dim, self.embedding_dim) * (1/np.sqrt(self.ff_dim))).astype(np.float32) # Weight second layer of shape (ff_dim, embedding_dim)
        self.B2 = np.zeros(self.embedding_dim, dtype=np.float32) # Bias second layer

        self.dW1 = np.zeros_like(self.W1)
        self.dB1 = np.zeros_like(self.B1)
        self.dW2 = np.zeros_like(self.W2)
        self.dB2 = np.zeros_like(self.B2)

        self.ff_input = embeddings.fwd(token_ids)
        self.hidden_layer = self.ff_input @ self.W1 + self.B1
        
        self.activated_layer = self.GELU(self.hidden_layer)

        self.output = self.activated_layer @ self.W2 + self.B2



    def GELU(self, x):
        """
        GELU activation function

        Args:
            x (array): array of vectors to go through activation function (3D matrix)

        Returns:
            array: activated layer from hidden layer
        """
        return 0.5 * x *(1+np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def ReLU(self, x):
        """Basically GELU but simpler

        Args:
            x (array): array of vectors to go through activation function (3D matrix)

        Returns:
            array: activated layer from hidden layer
        """
        return np.maximum(0, x)
    
    def fwd(self, x):
        """
        Performs the forward pass of the feed-forward network.

        Args:
            x (np.ndarray): Input array of shape (batch_size, embedding_dim).

        Returns:
            np.ndarray: Output array of shape (batch_size, embedding_dim).
        """
        self.ff_input = x
        self.hidden_layer = self.ff_input @ self.W1 + self.B1
        self.activated_layer = self.GELU(self.hidden_layer)
        self.output = self.activated_layer @ self.W2 + self.B2
        return self.output, self.hidden_layer, self.activated_layer

    def backward(self, dout, input_to_ffn, activated_layer, hidden_layer):
        """
        Performs the backward pass and calculates gradients.

        Args:
            dout (np.ndarray): Gradient of loss with respect to output, shape (batch_size, seq_len, embedding_dim).
            input_to_ffn (np.ndarray): Input to the forward pass of the feed-forward network,
                                       shape (batch_size, seq_len, embedding_dim).
            activated_layer (np.ndarray): Output of the first linear layer after activation,
                                          shape (batch_size, seq_len, ff_dim).
            hidden_layer (np.ndarray): Output of the first linear layer before activation,
                                       shape (batch_size, seq_len, ff_dim).

        Returns:
            np.ndarray: Gradient of loss with respect to input x, shape (batch_size, seq_len, embedding_dim).
        """
        # Gradient of output layer
        self.dW2 = np.einsum('bsf, bse -> fe', activated_layer, dout) # f: ff_dim, e: embedding_dim
        self.dB2 = np.sum(dout, axis=(0, 1))

        # Gradient through activation
        dactivated = dout @ self.W2.T
        dgelu = dactivated * (0.5 * (1 + np.tanh(np.sqrt(2/np.pi)*(hidden_layer + 0.044715 * hidden_layer**3))) + 
                             0.5 * hidden_layer * (1 - np.tanh(np.sqrt(2/np.pi)*(hidden_layer + 0.044715 * hidden_layer**3))**2) * 
                             np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * hidden_layer**2))

        self.dW1 = np.einsum('bse, bsf -> ef', input_to_ffn, dgelu) # e: embedding_dim, f: ff_dim
        self.dB1 = np.sum(dgelu, axis=(0, 1))

        # Gradient with respect to input
        dx = dgelu @ self.W1.T

        return dx
    
    def get_params_and_grads(self):
        return [
            {'value': self.W1, 'grad': self.dW1},
            {'value': self.B1, 'grad': self.dB1},
            {'value': self.W2, 'grad': self.dW2},
            {'value': self.B2, 'grad': self.dB2},
        ]

def main():

    with open('artifacts/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    embedding_layer = EmbeddingLayer(vocab_size=tokenizer.vocab_size)

    sample_texts = [
    "Hello World. My name is Albert Lungu",
    "What is your name?",
    "I like transformers",
]
    token_ids = [tokenizer.encode(text) for text in sample_texts]

    ff_class = FeedForward(token_ids, embedding_layer)
    # print(ff_class.hidden_layer)
#     print("Output shape: ", np.shape(ff_class.forward(ff_class.ff_input)))


if __name__ == "__main__":
    main()
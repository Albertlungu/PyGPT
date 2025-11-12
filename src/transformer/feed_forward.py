import numpy as np
import pickle
import sys
import os
import jax
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.embeddings.embeddings import EmbeddingLayer
from src.tokenizer.tokenizer_class import BPETokenizer 
from src.training.loss_function import CrossEntropyLoss

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
    

    def __init__(self, embeddings: EmbeddingLayer, ff_dim = 0):
        """
        Initializes the FeedForward network.

        Args:
            token_ids (list): List of token IDs representing input sequences.
            embeddings (EmbeddingLayer): An instance of EmbeddingLayer to convert token IDs to embeddings.

        Attributes:
            embedding_dim (int): Dimensionality of the embeddings.
            ff_dim (int): Dimensionality of the feed-forward hidden layer (4 times embedding_dim).
            W1 (jnp.ndarray): Weight matrix for the first linear layer of shape (embedding_dim, ff_dim).
            B1 (jnp.ndarray): Bias vector for the first linear layer of shape (ff_dim,).
            W2 (jnp.ndarray): Weight matrix for the second linear layer of shape (ff_dim, embedding_dim).
            B2 (jnp.ndarray): Bias vector for the second linear layer of shape (embedding_dim,).
            ff_input (jnp.ndarray): Input embeddings to the feed-forward network.
        """
        self.embedding_dim = EmbeddingLayer.default_embedding_dim
        self.ff_dim = ff_dim or self.embedding_dim * 4 # Feed Forward dimension

        self.key = jax.random.PRNGKey(0)

        k1, k2 = jax.random.split(self.key)

        self.cross_entropy = CrossEntropyLoss()
        self.loss_fn = self.cross_entropy.fwd

        # Layers 
        self.W1 = jax.random.normal(k1, (self.embedding_dim, self.ff_dim)) * (1 / jnp.sqrt(self.embedding_dim)) # Weight first layer of shape (embedding_dim, ff_dim)
        self.B1 = jnp.zeros(self.ff_dim)# Bias first layer
        self.W2 = jax.random.normal(k2, (self.ff_dim, self.embedding_dim)) * (1 / jnp.sqrt(self.ff_dim)) # Weight second layer of shape (ff_dim, embedding_dim)
        self.B2 = jnp.zeros(self.embedding_dim) # Bias second layer
            
    @staticmethod
    @jax.jit
    def GELU(x):
        """
        GELU activation function

        Args:
            x (array): array of vectors to go through activation function (3D matrix)

        Returns:
            array: activated layer from hidden layer
        """
        return 0.5 * x * (1+jnp.tanh(jnp.sqrt(2/jnp.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    @jax.jit
    def ReLU(x):
        """Basically GELU but simpler

        Args:
            x (array): array of vectors to go through activation function (3D matrix)

        Returns:
            array: activated layer from hidden layer
        """
        return jnp.maximum(0, x)
    
    @jax.jit
    def fwd(self, x):
        """
        Performs the forward pass of the feed-forward network.

        Args:
            x (jnp.ndarray): Input array of shape (batch_size, embedding_dim).

        Returns:
            jnp.ndarray: Output array of shape (batch_size, embedding_dim).
        """
        hidden = x @ self.W1 + self.B1
        activated = self.GELU(hidden)
        output = activated @ self.W2 + self.B2
        return output
    
    def compute_grads(self, x, target_ids):
        """
        Computes gradients of the mean squared error loss w.r.t. the weights and biases.

        Args:
            x (jnp.ndarray): Input embeddings (batch_size, embedding_dim)
            target (jnp.ndarray): Target embeddings of same shape

        Returns:
            dict: Gradients for W1, B1, W2, B2
        """
        def loss_fn(W1, B1, W2, B2):
            hidden = x @ W1 + B1
            activated = self.GELU(hidden)
            logits = activated @ W2 + B2
            return self.loss_fn(logits, target_ids)
        
        grads = jax.grad(loss_fn, argnums=(0,1,2,3))(self.W1, self.B1, self.W2, self.B2)

        return{
            'dW1': grads[0],
            'dB1': grads[1],
            'dW2': grads[2],
            'dB2': grads[3]
        }
    
    def get_params_and_grads(self, grads):
        return [
            {'value': self.W1, 'grad': grads['dW1']},
            {'value': self.B1, 'grad': grads['dB1']},
            {'value': self.W2, 'grad': grads['dW2']},
            {'value': self.B2, 'grad': grads['dB2']},
        ]

def main():
    pass


if __name__ == "__main__":
    main()
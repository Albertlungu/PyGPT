import pickle
import sys
import os
import jax
import jax.numpy as jnp
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.embeddings.embeddings import EmbeddingLayer
from src.training.loss_function import CrossEntropyLoss

class MultiHeadAttention:
    """
      Multi-head self-attention mechanism implemented with JAX for efficient autodiff and JIT compilation.

      This class implements the multi-head attention mechanism from "Attention is All You Need" (Vaswani et al., 2017).
      It splits the embedding dimension into multiple attention heads, allowing the model to jointly attend to
      information from different representation subspaces at different positions.
      
      Attributes:
          embedding_dim (int): Dimensionality of input embeddings (e.g., 256)
          num_heads (int): Number of parallel attention heads (default: 8)
          head_dim (int): Dimension per head (embedding_dim // num_heads)
          W_Q (jnp.ndarray): Query projection matrix, shape (embedding_dim, embedding_dim)
          W_K (jnp.ndarray): Key projection matrix, shape (embedding_dim, embedding_dim)
          W_V (jnp.ndarray): Value projection matrix, shape (embedding_dim, embedding_dim)
          W_O (jnp.ndarray): Output projection matrix, shape (embedding_dim, embedding_dim)

      """
    def __init__(self, embedding_layer: EmbeddingLayer, num_heads = 8):
        """
        Initializing MutliHeadAttention

        Args:
            embedding_layer (EmbeddingLayer): Instance of EmbeddingLayer class
            num_heads (int, optional): Number of heads for multi-head attention. Defaults to 8.
        """
        self.embedding_dim = embedding_layer.embedding_dim
        
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads

        key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.W_Q = jax.random.normal(k1, (self.embedding_dim, self.embedding_dim)) * 0.01
        self.W_K = jax.random.normal(k2, (self.embedding_dim, self.embedding_dim)) * 0.01
        self.W_V = jax.random.normal(k3, (self.embedding_dim, self.embedding_dim)) * 0.01
        self.W_O = jax.random.normal(k4, (self.embedding_dim, self.embedding_dim)) * 0.01

    @staticmethod
    @jax.jit
    def fwd(params, x, num_heads, head_dim, embedding_dim):
        """
        Pure function for jit computation

        Args:
            params (dict): dict with W_Q, W_K, W_V, W_O
            x (jnp.ndarray): Input embeddings (shape: batch, seq_len, embedding_dim)

        Return:
            jnp.ndarray: Final output in embeddings
        """

        # Project to query, key, value
        Q = x @ params['W_Q']
        K = x @ params['W_K']
        V = x @ params["W_V"]

        # Reshape for multi-head (batch, seq_len, num_heads, head_dim)
        batch_size, seq_len, _ = x.shape
        Q = Q.reshape(batch_size, seq_len, num_heads, head_dim) # See readme for what .reshape does
        K = K.reshape(batch_size, seq_len, num_heads, head_dim)
        V = V.reshape(batch_size, seq_len, num_heads, head_dim)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(0,2,1,3)
        K = K.transpose(0,2,1,3)
        V = V.transpose(0,2,1,3)

        # Scaled dot product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / jnp.sqrt(head_dim)

        # Casual mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len))) # See readme for what jnp.tril and jnp.ones do
        scores = jnp.where(mask == 0, -1e9, scores) # See readme for what jnp.where does

        # Softmax to find weights
        attn_weights = jax.nn.softmax(scores, axis=-1) # See readme for what axis=-1 is

        # Apply attention to values
        attn_output = attn_weights @ V

        # Combining the heads together to make final output
        attn_output = attn_output.transpose(0,2,1,3)
        attn_output = attn_output.reshape(batch_size, seq_len, embedding_dim)

        output = attn_output @ params["W_O"]

        return output
    
    def compute_grads(self, x, d_output):
        """
        Most efficient version of backprop using JAX's vjp

        Args:
            x: input (batch, seq_len, embedding_dim)
            d_output: gradient from next layer (batch, seq_len, embedding_dim)

        Returns:
            grads: dict of parameter gradients
            d_input: gradient w.r.t input
        """
        params = self.get_params()
        output, vjp_fn = jax.vjp(
            lambda p, x_: self.fwd(p, x_, self.num_heads, self.head_dim, self.embedding_dim), params, x)
        grads_params, d_input = vjp_fn(d_output)
        return grads_params, d_input
    
    def get_params_and_grads(self, grads = None):
        """
        Return params and grads in the format Trainer expects.

        Args:
            grads (dict, optional): Dictionary of gradients. Defaults to None.

        Returns:
            list of dicts with 'value' and 'grad' keys
        """
        if grads is None:
            grads = {
                "W_Q": jnp.zeros_like(self.W_Q),
                "W_K": jnp.zeros_like(self.W_K),
                "W_V": jnp.zeros_like(self.W_V),
                "W_O": jnp.zeros_like(self.W_O)
            }
        return [
            {'value': self.W_Q, 'grad': grads['W_Q']},
            {'value': self.W_K, 'grad': grads['W_K']},
            {'value': self.W_V, 'grad': grads['W_V']},
            {'value': self.W_O, 'grad': grads['W_O']},
        ]
    
    def get_params(self):
        return {
            "W_Q": self.W_Q,
            "W_K": self.W_K,
            "W_V": self.W_V,
            "W_O": self.W_O
        }
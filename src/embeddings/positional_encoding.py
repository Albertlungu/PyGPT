"""
Positional encoding for transformer models.

This module provides sinusoidal positional encodings to give the model
information about token positions in sequences.
"""
import jax # pylint: disable=no-member
import jax.numpy as jnp # pylint: disable=no-member
import numpy as np

class PositionalEncoding:
    """
    Sinusoidal positional encoding for transformer architectures.

    Generates position-dependent patterns using sine and cosine functions
    to encode sequence position information.
    """
    def __init__(self, embedding_dim:int, max_seq_length=256) -> None:
        """
        Initialize the positional encoding generator.

        Args:
            embedding_dim (int): Embedding dimension.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 256.
        """
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim

    def _create_positional_encoding(self, n=10000) -> jnp.array:
        """
        Create sinusoidal positional encodings.

        Args:
            n (int, optional): Frequency parameter for encoding. Defaults to 10000.

        Returns:
            jnp.ndarray: Positional encoding matrix of shape (max_seq_length, embedding_dim).
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
        pos = jnp.arange(L)[:, None]
        i = jnp.arange(d)[None, :]
        angle_rates = 1 / jnp.power(n, (2 * (i//2)) / d)
        P = pos * angle_rates
        P = P.at[:, 0::2].set(jnp.sin(P[:, 0::2]))
        P = P.at[:, 1::2].set(jnp.cos(P[:, 1::2]))
        return P

import numpy as np
import jax
import jax.numpy as jnp

class PositionalEncoding:
    def __init__(self, embedding_dim:int, max_seq_length=256) -> None:
        """
        Initializing the positional encoding class

        Args:
            embedding_dim (int): Embedding dimension
            max_seq_length (int, optional): Maximum sequence length. Defaults to 256.
        """
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
          
    def _create_positional_encoding(self, n=10000) -> jnp.array:
        # making variables in accordance to understanding_positional_encoding.md
        """
        Creates positional encoding for the given embedding layer.

        Parameters:
            n (int, optional): the maximum sequence length. Defaults to 10000.

        Returns:
            jnp.Array: the positional encoding function, that describes the position of word in a given input
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

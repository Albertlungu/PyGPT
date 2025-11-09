import numpy as np

class PositionalEncoding:
    def __init__(self, embedding_dim, max_seq_length = 512):
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
          
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
        pos = np.arange(L, dtype=np.float32)[:, np.newaxis]
        i = np.arange(d, dtype=np.float32)[np.newaxis, :]
        angle_rates = 1 / np.power(n, (2 * (i//2)) / d)
        P = pos * angle_rates
        P[:, 0::2] = np.sin(P[:, 0::2])
        P[:, 1::2] = np.cos(P[:, 1::2])
        return P.astype(np.float32)
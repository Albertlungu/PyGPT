import numpy as np
import pickle
from embeddings import EmbeddingLayer
from attention import Attention
from feed_forward import FeedForward

class TransformerBlock:
    """
    Represents a single transformer block, including attention and feedforward layers.

    Attributes:
        token_ids (list or array): Token indices for the input sequence.
        embedding_layer (EmbeddingLayer): Embedding layer used to convert token_ids into embeddings.
        attention_layer (Attention): The attention mechanism for the block.
        ffn (FeedForward): Feedforward network for the block.
        gamma_1, beta_1: Learnable parameters for first layer normalization.
        gamma_2, beta_2: Learnable parameters for second layer normalization.
        attention_output: Output from the attention layer.
    """
    
    def __init__(self, token_ids, embedding_layer: EmbeddingLayer):
        """
        Initializing instance variables for the TransformerBlock class

        Args:
            token_ids (list): IDs of input tokens given to model (padding is already applied in FeedForward layer)
            embedding_layer (EmbeddingLayer): EmbeddingLayer class. Takes no arguments

        Returns:
            _type_: _description_
        """
        self.embedding_layer = embedding_layer
        self.token_ids = token_ids
        
        self.attention_layer = Attention(token_ids, embedding_layer)
        self.ffn = FeedForward(embedding_layer, token_ids)

        self.embedding_dim = embedding_layer.embedding_dim

        # First normalization: layer normalization 1 scale. Makes learnable scaling after normalization
        self.gamma_1 = np.ones(self.embedding_dim)
        # First normalization: layer norm shift param. Makes learnable offset after normalization
        self.beta_1 = np.zeros(self.embedding_dim)

        self.gamma_2 = np.ones(self.embedding_dim)
        self.beta_2 = np.zeros(self.embedding_dim)

        self.attention_output = self.attention_layer.fwd()


        @staticmethod
        def layer_norm(self, x, gamma, beta, epsilon = 1e-5):
            """
            Normalizes each token's features to have zero mean and unit variance, then applies learnable scale (gamma) and shift (beta).
                mean: The average value of the features for each token, computed along the last axis.
                variance: The measure of spread of the features for each token, computed as the average squared deviation from the mean.
                normalized: The features after subtracting the mean and dividing by the standard deviation, resulting in zero mean and unit variance.

            
            Args:
                x (3D tensor): the input tensor for the TransformerBlock. a.k.a the output from the fwd of the attention. Shape: (batch, seq_len, embedding_dim)
                gamma (vector): A learnable scaling parameter applied after normalization to allow the model to adjust the normalized output.
                beta (vector): A learnable shifting parameter applied after scaling to allow the model to offset the normalized output.
                epsilon (float, optional): A small constant added to the variance for numerical stability to avoid division by zero. Defaults to 1e-5.

            Returns:
                3D tensor: A tensor of the same shape as the input (batch, seq_len, embedding_dim), where each token's features have been normalized to zero mean and unit variance, then scaled by `gamma` and shifted by `beta`.
            """

            mean = np.mean(x, axis = -1 , keepdims=True)
            mean = np.var(x, axis = -1, keepdims=True)

            normalized = (self.attention_output - mean) / np.sqrt(mean + epsilon)

            output = gamma * normalized + beta

            return output
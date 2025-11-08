import numpy as np
import pickle
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.embeddings.embeddings import EmbeddingLayer
from src.transformer.single_head_attention import Attention
from feed_forward import FeedForward
from src.tokenizer.tokenizer_class import BPETokenizer

class TransformerBlock:
    """
    Represents a single transformer block, including attention and feedforward layers.

    Attributes:
        token_ids (list or array): Token indices for the input sequence.
        embedding_layer (EmbeddingLayer): Embedding layer used to convert token_ids into embeddings.
        attention_layer (Attention): The attention mechanism for the block.
        embedding_dim (integer): The size of the vector used to represent each token in the model.
        input_embeddings (3D tensor): The np.ndarray that represents my input tokens as a vector quantity.
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
        self.ffn = FeedForward(token_ids, embedding_layer)

        self.embedding_dim = embedding_layer.embedding_dim
        self.input_embeddings = self.embedding_layer.forward(self.token_ids)

        # First normalization: layer normalization 1 scale. Makes learnable scaling after normalization
        self.gamma_1 = np.ones(self.embedding_dim)
        # First normalization: layer norm shift param. Makes learnable offset after normalization
        self.beta_1 = np.zeros(self.embedding_dim)

        self.gamma_2 = np.ones(self.embedding_dim)
        self.beta_2 = np.zeros(self.embedding_dim)

    @staticmethod
    def layer_norm(x, gamma, beta, epsilon = 1e-5):
            """
            Normalizes each token's features to have mean = 0 and variance = 1, then applies learnable scale (gamma) and shift (beta).
                mean: The average value of the features for each token, computed along the last axis.
                variance: The measure of spread of the features for each token, computed as the average squared deviation from the mean.
                normalized: The features after subtracting the mean and dividing by the standard deviation, resulting in zero mean and unit variance.
                token's feature: one component of the vector that represents that token

            
            Args:
                x (3D tensor): the input tensor for the TransformerBlock. a.k.a the output from the fwd of the attention. Shape: (batch, seq_len, embedding_dim)
                gamma (vector): A learnable scaling parameter applied after normalization to allow the model to adjust the normalized output.
                beta (vector): A learnable shifting parameter applied after scaling to allow the model to offset the normalized output.
                epsilon (float, optional): A small constant added to the variance for numerical stability to avoid division by zero. Defaults to 1e-5.

            Returns:
                3D tensor: A tensor of the same shape as the input (batch, seq_len, embedding_dim), where each token's features have been normalized to zero mean and unit variance, then scaled by `gamma` and shifted by `beta`.
            """

            mean = np.mean(x, axis = -1 , keepdims=True)
            var = np.var(x, axis = -1, keepdims=True)

            normalized = (x - mean) / np.sqrt(var + epsilon)

            output = gamma * normalized + beta

            return output
        
    def fwd(self):
        """
        Performs a forward pass through the Transformer Block

        Steps:
        1. Sublayer 1: Attention
            - Stores the input embeddings as `residual_1`.
            - Applies layer normalization to the input (`ln1_out`).
            - Passes normalized input through the attention layer.
            - Adds the residual connection to the attention output (`after_attention`).

        2. Sublayer 2: FeedForward
            - Stores the output of the attention sublayer as `residual_2`.
            - Applies layer normalization to the attention output (`ln2_out`).
            - Passes normalized input through the feedforward network.
            - Adds the residual connection to the feedforward output (`final_output`).

        Returns:
            3D tensor (np.ndarray): The final output tensor with the same shape as the input 
                `(batch_size, sequence_length, embedding_dimension)`
        """

        # ===== Sublayer 1 =====
        self.residual_1 = self.input_embeddings # Copy of embeddings to be added to attention_otuput
        self.ln1_out = self.layer_norm(self.input_embeddings, self.gamma_1, self.beta_1)
        self.attention_output = self.attention_layer.fwd(self.ln1_out)
        self.after_attention = self.residual_1 + self.attention_output
        # print("After attention shape: ", np.shape(self.after_attention))

        # ===== Sublayer 2 =====
        self.residual_2 = self.after_attention # Copy of after_attention, taken from sublayer 1
        self.ln2_out = self.layer_norm(self.after_attention, self.gamma_2, self.beta_2)
        self.ff_output = self.ffn.fwd(self.ln2_out)
        self.final_output = self.residual_2 + self.ff_output

        return self.final_output



def main():

    embedding_layer = EmbeddingLayer()

    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    sample_texts = [
        "Hello World. My name is Albert Lungu",
        "What is your name?",
        "I like LLMs"
    ] # Batch size = 3, seq_len = 12, embedding_dim = 256 probably

    token_ids = [tokenizer.encode(text) for text in sample_texts]

    transformer_block = TransformerBlock(token_ids=token_ids, embedding_layer=embedding_layer)

    transformer_block.fwd()

if __name__ == "__main__":
    main()
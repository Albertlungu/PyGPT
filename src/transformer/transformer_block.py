import numpy as np
import pickle
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.embeddings.embeddings import EmbeddingLayer
from src.transformer.single_head_attention import Attention
from src.transformer.feed_forward import FeedForward
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
        self.input_embeddings = self.embedding_layer.fwd(self.token_ids)

        # First normalization: layer normalization 1 scale. Makes learnable scaling after normalization
        self.gamma_1 = np.ones(self.embedding_dim, dtype=np.float32)
        # First normalization: layer norm shift param. Makes learnable offset after normalization
        self.beta_1 = np.zeros(self.embedding_dim, dtype=np.float32)

        self.gamma_2 = np.ones(self.embedding_dim, dtype=np.float32)
        self.beta_2 = np.zeros(self.embedding_dim, dtype=np.float32)


        self.d_gamma_1 = np.zeros_like(self.gamma_1, dtype=np.float32)
        self.d_beta_1  = np.zeros_like(self.beta_1, dtype=np.float32)
        self.d_gamma_2 = np.zeros_like(self.gamma_2, dtype=np.float32)
        self.d_beta_2  = np.zeros_like(self.beta_2, dtype=np.float32)

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
        self.attention_output, self.attention_input_fwd, self.attention_output_fwd, self.attention_V_fwd, self.attention_weights_fwd, self.attention_K_fwd = self.attention_layer.fwd(self.ln1_out)
        self.after_attention = self.residual_1 + self.attention_output
        # print("After attention shape: ", np.shape(self.after_attention))

        # ===== Sublayer 2 =====
        self.residual_2 = self.after_attention # Copy of after_attention, taken from sublayer 1
        self.ln2_out = self.layer_norm(self.after_attention, self.gamma_2, self.beta_2)
        self.ff_input_to_fwd = self.ln2_out # Store input to FFN for backward pass
        self.ff_output, self.ffn_hidden_layer, self.ffn_activated_layer = self.ffn.fwd(self.ff_input_to_fwd)
        self.final_output = self.residual_2 + self.ff_output

        return self.final_output

    def backward(self, d_out):
        """
        Backward pass through the TransformerBlock.

        Args:
            d_out (3D tensor): Gradient of loss with respect to the final output
                shape: (batch_size, seq_len, embedding_dim)

        Steps:
            1. Backprop through residual + feedforward sublayer.
            2. Backprop through second layer normalization.
            3. Backprop through attention sublayer and first layer normalization.
            4. Store gradients for gamma and beta parameters.
        """

        # ===== Sublayer 2: FeedForward + Residual =====
        # Gradient w.r.t residual connection
        d_residual2 = d_out  # Gradient flows to residual
        d_ff_output = d_out  # Gradient also flows to feedforward output

        # Backprop through feedforward network
        batch_size_dout = d_ff_output.shape[0]
        truncated_ff_input = self.ff_input_to_fwd[:batch_size_dout, :, :]
        truncated_activated_layer = self.ffn_activated_layer[:batch_size_dout, :, :]
        truncated_hidden_layer = self.ffn_hidden_layer[:batch_size_dout, :, :]
        d_ln2_out = self.ffn.backward(d_ff_output, truncated_ff_input, truncated_activated_layer, truncated_hidden_layer)  # shape: (batch, seq_len, embedding_dim)

        # Add gradient from residual
        d_ln2_out += d_residual2  # total gradient to ln2 input

        # ===== LayerNorm 2 backward =====
        batch_size_d_ln2_out = d_ln2_out.shape[0]
        truncated_after_attention = self.after_attention[:batch_size_d_ln2_out, :, :]
        d_after_attention, d_gamma2, d_beta2 = self.layer_norm_backward(d_ln2_out, truncated_after_attention, self.gamma_2, self.beta_2)
        # Store gradients for gamma and beta
        self.d_gamma_2 = d_gamma2
        self.d_beta_2 = d_beta2

        # ===== Sublayer 1: Attention + Residual =====
        # Gradient flows through residual
        d_residual1 = d_after_attention
        # Backprop through attention
        batch_size_d_after_attention = d_after_attention.shape[0]
        truncated_attention_input_fwd = self.attention_input_fwd[:batch_size_d_after_attention, :, :]
        truncated_attention_output_fwd = self.attention_output_fwd[:batch_size_d_after_attention, :, :]
        truncated_attention_V_fwd = self.attention_V_fwd[:batch_size_d_after_attention, :, :]
        truncated_attention_weights_fwd = self.attention_weights_fwd[:batch_size_d_after_attention, :, :]
        truncated_attention_K_fwd = self.attention_K_fwd[:batch_size_d_after_attention, :, :]
        d_ln1_out = self.attention_layer.backward(d_after_attention, truncated_attention_input_fwd, truncated_attention_output_fwd, truncated_attention_V_fwd, truncated_attention_weights_fwd, truncated_attention_K_fwd)  # shape: (batch, seq_len, embedding_dim)

        # Add gradient from residual
        d_ln1_out += d_residual1

        # ===== LayerNorm 1 backward =====
        batch_size_d_ln1_out = d_ln1_out.shape[0]
        truncated_input_embeddings = self.input_embeddings[:batch_size_d_ln1_out, :, :]
        d_input_embeddings, d_gamma1, d_beta1 = self.layer_norm_backward(d_ln1_out, truncated_input_embeddings, self.gamma_1, self.beta_1)
        self.d_gamma_1 = d_gamma1
        self.d_beta_1 = d_beta1

        # ===== Backprop to embeddings =====
        self.embedding_layer.backward(d_input_embeddings)  # propagate gradients to embeddings

    def layer_norm_backward(self, d_out, x, gamma, beta, epsilon=1e-5):
        """
        Backprop through layer normalization.

        Args:
            d_out: Gradient of loss w.r.t output of layer norm
            x: Input to layer norm
            gamma, beta: parameters
            epsilon: small value for stability

        Returns:
            d_x: gradient w.r.t input x
            d_gamma: gradient w.r.t gamma
            d_beta: gradient w.r.t beta
        """
        N = x.shape[-1]
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + epsilon)
        x_norm = (x - mean) / std

        # Gradients w.r.t gamma and beta
        d_gamma = np.sum(d_out * x_norm, axis=(0, 1))
        d_beta = np.sum(d_out, axis=(0, 1))

        # Gradient w.r.t normalized input
        dx_norm = d_out * gamma

        # Backprop through normalization
        d_var = np.sum(dx_norm * (x - mean) * -0.5 * (var + epsilon)**(-1.5), axis=-1, keepdims=True)
        d_mean = np.sum(-dx_norm / std, axis=-1, keepdims=True) + d_var * np.mean(-2 * (x - mean), axis=-1, keepdims=True)
        d_x = dx_norm / std + d_var * 2 * (x - mean) / N + d_mean / N

        return d_x, d_gamma, d_beta
    
    def get_params_and_grads(self):
        params = []

        # Collect params from attention and feedforward submodules
        params.extend(self.attention_layer.get_params_and_grads())
        params.extend(self.ffn.get_params_and_grads())

        # Include gamma and beta from both layer norms
        params.extend([
            {'value': self.gamma_1, 'grad': self.d_gamma_1},
            {'value': self.beta_1,  'grad': self.d_beta_1},
            {'value': self.gamma_2, 'grad': self.d_gamma_2},
            {'value': self.beta_2,  'grad': self.d_beta_2},
        ])

        return params


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
import jax
import jax.numpy as jnp
import pickle
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.embeddings.embeddings import EmbeddingLayer
from src.transformer.multi_head_attention import MultiHeadAttention
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
    
    def __init__(self, embedding_layer: EmbeddingLayer, num_heads = 8):
        """
        Initializing instance variables for the TransformerBlock class

        Args:
            token_ids (list): IDs of input tokens given to model (padding is already applied in FeedForward layer)
            embedding_layer (EmbeddingLayer): EmbeddingLayer class. Takes no arguments

        Returns:
            _type_: _description_
        """

        self.embedding_dim = embedding_layer.embedding_dim
        self.num_heads = num_heads

        self.attention_layer = MultiHeadAttention(embedding_layer, num_heads)
        self.ffn = FeedForward(embedding_layer)

        self.gamma_1 = jnp.ones((self.embedding_dim,))
        self.beta_1 = jnp.zeros((self.embedding_dim,))
        self.gamma_2 = jnp.ones((self.embedding_dim,))
        self.beta_2 = jnp.zeros((self.embedding_dim,))

    @staticmethod
    def layer_norm(x, gamma, beta, epsilon = 1e-5):
        """
        Layer normalization - normalizes across the feature dimension.

        Computes: output = gamma * (x - mean) / sqrt(variance + epsilon) + beta

        Args:
            x (jnp.ndarray): Input tensor (batch, seq_len, embedding_dim)
            gamma (jnp.ndarray): Scale parameter (embedding_dim,)
            beta (jnp.ndarray): Shift parameter (embedding_dim,)
            epsilon (float): Small constant for numerical stability

        Returns:
            jnp.ndarray: Normalized output, same shape as input
        """

        mean = jnp.mean(x, axis=-1 , keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(var + epsilon)
        output = gamma * normalized + beta

        return output
    
    @staticmethod
    def fwd(params, x, num_heads, head_dim, embedding_dim):
        """
        Forward pass through transformer block (pure function for JIT).

        Flow:
            1. LayerNorm → MultiHeadAttention → Add residual
            2. LayerNorm → FeedForward → Add residual

        Args:
            params (dict): Contains all parameters:
                - 'attn': attention parameters
                - 'ffn': feed-forward parameters
                - 'gamma_1', 'beta_1': first LayerNorm params
                - 'gamma_2', 'beta_2': second LayerNorm params
            x (jnp.ndarray): Input embeddings (batch, seq_len, embedding_dim)
            num_heads (int): Number of attention heads
            head_dim (int): Dimension per head
            embedding_dim (int): Total embedding dimension

        Returns:
            jnp.ndarray: Output (batch, seq_len, embedding_dim)
          """

        # ==== Sublayer 1 ====
        residual_1 = x
        ln1_out = TransformerBlock.layer_norm(x, params['gamma_1'], params['beta_1'])

        attn_output = MultiHeadAttention.fwd(
            params['attn'],
            ln1_out,
            num_heads,
            head_dim,
            embedding_dim
        )
        after_attention = residual_1 + attn_output

        # ==== Sublayer 2 ====
        residual_2 = after_attention
        ln2_out = TransformerBlock.layer_norm(
            after_attention,
            params['gamma_2'],
            params['beta_2']
        )

        ff_output = FeedForward.fwd(params['ffn'], ln2_out)
        final_output = residual_2 + ff_output

        return final_output
    
    @staticmethod
    @jax.jit
    def fwd_with_cache(params, x, num_heads, head_dim, embedding_dim, past_kv=None):
        """
        Forward pass with KV-cache

        Args:
            params (dict): Block params
            x (jnp.jnparray): new token embeddings, shape: [batch, 1, embedding_dim]
            num_heads (int): Number of attention heads
            head_dim (int): Dimension of each head
            embedding_dim (int): Embedding dimension
            past_kv (jnp.jnparray, optional): Cached K and V from the layer's other calls. Defaults to None.
        
        Returns:
            output: (batch, 1, embedding_dim)
            new_kv: Updated (K, V) cache
        """
        res1 = x
        ln1_out = TransformerBlock.layer_norm(x, params['gamma_1'], params['beta_1'])

        attn_out, new_kv = MultiHeadAttention.fwd_with_cache(
            params['attn'],
            ln1_out,
            num_heads,
            head_dim,
            embedding_dim,
            past_kv=past_kv
        )
        after_attention = res1 + attn_out

        res2 = after_attention
        ln2_out = TransformerBlock.layer_norm(after_attention, params['gamma_2'], params['beta_2'])

        ff_out = FeedForward.fwd(params['ffn'], ln2_out)
        final_out = res2 + ff_out

        return final_out, new_kv

    def get_params(self):
        """
        Get all parameters as a dictionary for JAX functions

        Returns:
            dict: All trainable params
        """
        return {
            'attn': self.attention_layer.get_params(),
            'ffn':{
                'W1': self.ffn.W1,
                'B1': self.ffn.B1,
                'W2': self.ffn.W2,
                'B2': self.ffn.B2
            },
            'gamma_1': self.gamma_1,
            'beta_1': self.beta_1,
            'gamma_2': self.gamma_2,
            'beta_2': self.beta_2
        }
    
    def compute_grads(self, x, d_output):
        """
        Compute gradients using JAX autodiff

        Args:
            x (jnp.ndarray): Input to this block (batch, seq_len, embedding_dim)
            d_output (jnp.ndarray): Gradient from next layer/transformer block 
        
        Returns:
            tuple: (grads_dict, d_input)
                - grads_dict: Gradients for all parameters
                - d_input: Gradient w.r.t (for previous block)
        """
        params = self.get_params()

        output, vjp_fn = jax.vjp(
            lambda p, x_: self.fwd(
                p, x_, self.num_heads,
                self.embedding_dim // self.num_heads,
                self.embedding_dim
            ),
            params, x
        )

        grads_params, d_input = vjp_fn(d_output)
        return grads_params, d_input
    
    def get_params_and_grads(self, grads=None):
        """
        Return params and grads in Trainer format

        Args:
            grads (dict, optional): Gradient dictionary. Defaults to None.

        Returns:
            list: List of {'value': param, 'grad': grad} dicts
        """
        
        if grads is None:
            grads = {
                'attn': {k: jnp.zeros_like(v) for k,v in self.attention_layer.get_params().items()},
                'ffn':{
                    'W1': jnp.zeros_like(self.ffn.W1),
                    'B1': jnp.zeros_like(self.ffn.B1),
                    'W2': jnp.zeros_like(self.ffn.W2),
                    'B2': jnp.zeros_like(self.ffn.B2)
                },
                'gamma_1': jnp.zeros_like(self.gamma_1),
                'beta_1': jnp.zeros_like(self.beta_1),
                'gamma_2': jnp.zeros_like(self.gamma_2),
                'beta_2': jnp.zeros_like(self.beta_2)
            }
        result = []

        for key in ['W_Q', 'W_K', "W_V", "W_O"]:
            result.append({
                'value': self.attention_layer.get_params()[key],
                'grad': grads['attn'][key]
            })
        
        result.extend([
            {'value': self.ffn.W1, 'grad': grads['ffn']['W1']},
            {'value': self.ffn.B1, 'grad': grads['ffn']['B1']},
            {'value': self.ffn.W2, 'grad': grads['ffn']['W2']},
            {'value': self.ffn.B2, 'grad': grads['ffn']['B2']}
        ])

        result.extend([
            {'value': self.gamma_1, 'grad': grads['gamma_1']},
            {'value': self.beta_1, 'grad': grads['beta_1']},
            {'value': self.gamma_2, 'grad': grads['gamma_2']},
            {'value': self.beta_2, 'grad': grads['beta_2']}
        ])

        return result


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
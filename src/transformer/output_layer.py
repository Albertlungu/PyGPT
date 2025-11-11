import pickle
import sys, os
import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.embeddings.embeddings import EmbeddingLayer
from src.tokenizer.tokenizer_class import BPETokenizer
from src.transformer.transformer_block import TransformerBlock


class OutputLayer:
    def __init__(self, embedding_layer: EmbeddingLayer):
        """
        Initializes instance attributes for OutputLayer class.

        Args:
            embedding_layer (EmbeddingLayer): EmbeddingLayer class.
        """
        self.embedding_layer = embedding_layer
        self.embedding_dim = self.embedding_layer.embedding_dim
        self.vocab_size = embedding_layer.vocab_size

        self.W_out = embedding_layer.embeddings.T # Weight matrix shape: (embedding_dim, vocab_size)
        # self.W_out = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01
        self.b_out = jnp.zeros(self.vocab_size) # Bias vector
    @staticmethod
    @jax.jit
    def fwd(params, transformer_output):
        """

        Args:
            transformer_output (3D Tensor): Output from last transformer block
                Shape: (batch_size, seq_len, embedding_dim)

        Returns:
            3D Tensor: Logits over vocabulary,
                Shape: (batch_size, seq_len, vocab_size)
        """
        logits = transformer_output @ params['W_out'] + params['b_out']
        # print(np.shape(self.logits))
        return logits
    
    def get_params(self):
        return {
            'W_out': self.W_out,
            'b_out': self.b_out
        }
    
    def compute_grads(self, transformer_output, d_output):
        params = self.get_params()
        output, vjp_fn = jax.vjp(
            lambda p, x: self.fwd(p, x),
            params,
            transformer_output
        )
        grads_params, d_input = vjp_fn(d_output)
        return grads_params, d_input
    
    def get_params_and_grads(self):
        if grads is None:
              grads = {
                  'W_out': jnp.zeros_like(self.W_out),
                  'b_out': jnp.zeros_like(self.b_out)
              }

        return [
            {'value': self.W_out, 'grad': grads['W_out']},
            {'value': self.b_out, 'grad': grads['b_out']}
        ]

    def predict_next_token(self, transformer_output, temperature = 1.0):
        """
        Samples the next token from the model's predictions.

        Args:
            transformer_output (np.ndarray): Output from last TransformerBlock
                                            shape: (batch_size, seq_len, embedding_dim)
            temperature (float): Sampling temperature (default 1.0)
                                Higher = more random, Lower = more deterministic

        Returns:
            np.ndarray: Predicted token IDs
                        shape: (batch_size,) - one prediction per sequence
        """
        params = self.get_params()
        logits = self.fwd(params, transformer_output)[:, -1, :]
        scaled_logits = logits / temperature
        probs = jax.nn.softmax(scaled_logits)
        predicted_tokens = jnp.argmax(probs, axis = -1)
        return predicted_tokens
        
    
def main():

    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    sample_texts = [
        "Hello World. My name is Albert Lungu",
        "What is your name?",
        "I like LLMs"
    ] # Batch size: 3, Seq_len = 12

    token_ids = [tokenizer.encode(text) for text in sample_texts]

    embedding_layer = EmbeddingLayer()
    output_layer = OutputLayer(embedding_layer)
    transformer_block = TransformerBlock(token_ids, embedding_layer)
    transformer_output = transformer_block.fwd()

    output_fwd = output_layer.fwd(transformer_output)


if __name__ == "__main__":
    main()
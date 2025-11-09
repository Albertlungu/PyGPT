import numpy as np
import pickle
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.embeddings.embeddings import EmbeddingLayer
from src.transformer.single_head_attention import Attention
from src.transformer.feed_forward import FeedForward
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
        self.b_out = np.zeros(self.vocab_size) # Bias vector

    def fwd(self, transformer_output):
        """

        Args:
            transformer_output (3D Tensor): Output from last transformer block
                Shape: (batch_size, seq_len, embedding_dim)

        Returns:
            3D Tensor: Logits over vocabulary,
                Shape: (batch_size, seq_len, vocab_size)
        """
        self.logits = transformer_output @ self.W_out + self.b_out
        # print(np.shape(self.logits))
        return self.logits
    
    def backward(self, d_out):
        """
        Backward pass through output layer.

        Args:
            d_out: Gradient of loss w.r.t output logits (batch, seq_len, vocab_size)

        Returns:
            Gradient w.r.t input x (batch, seq_len, embedding_dim)
        """
        # Gradients w.r.t weights and bias
        # input: (batch, seq_len, embedding_dim), d_out: (batch, seq_len, vocab_size)
        self.dW = np.einsum('bse, bsv -> ev', self.input, d_out)  # sum over batch & seq_len
        self.db = np.sum(d_out, axis=(0, 1), keepdims=True)       # sum over batch & seq_len

        # Gradient w.r.t input to pass to previous layer
        d_input = d_out @ self.W.T  # shape: (batch, seq_len, embedding_dim)
        return d_input
    
    @staticmethod
    def softmax(logits):
        """
        Calculates probabilities given a 3D tensor of logits

        Args:
            logits (np.ndarray): unnormalized output of the transformer_block

        Returns:
            np.ndarray: probabilities taken from logits through softmax function
        """
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs

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
        logits = self.fwd(transformer_output)[:, -1, :]
        scaled_logits = logits/temperature
        probs = self.softmax(scaled_logits)
        predicted_tokens = np.argmax(probs, axis = -1)
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

    print(np.shape(output_fwd)) # Output: (3, 12, 5000)
    print(output_fwd)
    print(np.shape(output_layer.softmax(output_fwd)))
    print(output_layer.softmax(output_fwd))


if __name__ == "__main__":
    main()
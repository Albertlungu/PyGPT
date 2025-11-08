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
        self.embedding_layer = embedding_layer
        self.embedding_dim = self.embedding_layer.embedding_dim
        self.vocab_size = embedding_layer.vocab_size

        self.W_out = embedding_layer.embeddings # Weight matrix
        self.b_out = np.zeros(self.vocab_size) # Bias vector

    def fwd(self, transformer_output):
        """

        Args:
            transformer_output (3D Tensor): Output from last transformer block
                Shape: (batch_size, seq_len, embedding_dim)

        Returns:
            3D Tensor: Logits over vocabulary,
                Shape: (batch_size, seq_len, embedding_dim)
        """
        logits = transformer_output @ self.W_out + self.b_out
        return logits
    
    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs
        
    
    
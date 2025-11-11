import numpy as np
import sys, os
import jax
import jax.numpy as jnp
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.transformer.output_layer import OutputLayer
from src.embeddings.embeddings import EmbeddingLayer
from src.transformer.transformer_block import TransformerBlock

class CrossEntropyLoss:
    """
    Function that tells the model how wrong its predictions are. 
    Cross-entropy measures the difference between the model's prediction (from output_layer) and the true next token.
    """
    def __init__(self, ignore_index = None, reduction = 'mean'):
        """
        Initializing CrossEntropyLoss instance attributes

        Args:
            ignore_index (int, optional): Which index to ignore. Defaults to None.
            reduction (str, optional): Type of reduction, 'sum' or 'mean'. Defaults to 'mean'.
        """
        self.ignore_index = ignore_index 
        self.reduction = reduction
        self.grad_fn = jax.grad(self.fwd)

    @staticmethod
    @jax.jit
    def fwd(logits, targets, reduction = 'mean', ignore_index = None):
        """
        Forward pass for the cross entropy loss function. Calculates the actual loss for each token.

        Args:
            logits (jnp.ndarray): Output of the OutputLayer class.
            targets (jnp.ndarray): The true "next-token" index that the model is supposed to predict. Shape: (batch, seq_len)

        Returns:
            jnp.float64: Loss of my model I guess

        Raises:
            TypeError
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        targets = jnp.clip(targets, 0, vocab_size - 1)

        selected_log_probs = jnp.take_along_axis(
            log_probs, targets[..., None], axis=-1
        )[...,0]


        if ignore_index is not None:
            mask = (targets != ignore_index).astype(jnp.float32)
        else:
            mask = jnp.ones_like(targets, dtype=jnp.float32)

        neg_log_prob = -selected_log_probs * mask

        if reduction == 'mean':
            loss = neg_log_prob.sum()/mask.sum()
        elif reduction == 'sum':
            loss = neg_log_prob.sum()
        else: 
            raise TypeError("Please enter either 'mean' or 'sum' as a reduction")
        
        return loss
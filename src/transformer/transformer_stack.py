import jax
import jax.numpy as jnp
from src.transformer.transformer_block import TransformerBlock
from src.embeddings.embeddings import EmbeddingLayer

class TransformerStack:
    """
    Stack of multiple transformer blocks for deep transformer architecture.

    Implements a sequence of transformer blocks where the output of one block
    feeds into the input of the next. More blocks = more model capacity and
    ability to learn complex patterns.

    Attributes:
        num_blocks (int): Number of stacked transformer blocks
        blocks (list): List of TransformerBlock instances
        embedding_dim (int): Dimension of embeddings
        num_heads (int): Number of attention heads per block
      """
    def __init__(self, embedding_layer: EmbeddingLayer, num_blocks=8, num_heads=8):
        """
        Initialize stack of transformer blocks.

        Args:
            embedding_layer (EmbeddingLayer): Embedding layer instance
            num_blocks (int): Number of blocks to stack (default: 4)
            num_heads (int): Number of attention heads per block (default: 8)
        """
        self.num_blocks = num_blocks
        self.embedding_dim = embedding_layer.embedding_dim
        self.num_heads = num_heads

        self.blocks = [
            TransformerBlock(embedding_layer, num_heads)
            for _ in range(num_blocks)
        ]
    
    def fwd(self, x):
        """
        Forward pass through all stacked blocks

        Args:
            x (jnp.ndarray): Input embeddings (batch, seq_len, embedding_dim)

        Returns:
            jnp.ndarray: Output after passing through all blocks
        """

        output = x
        for block in self.blocks:
            params = block.get_params()
            head_dim = self.embedding_dim // self.num_heads
            output = TransformerBlock.fwd(
                params, output, self.num_heads, head_dim, self.embedding_dim
            )
        return output

    def compute_grads(self, x, d_output):
        """
        Backpropagate through all blocks to compute gradients.

        Uses reverse-mode autodiff: start from output, go backwards.

        Args:
            x (jnp.ndarray): Original input to the stack
            d_output (jnp.ndarray): Gradient from loss

        Returns:
            list: List of gradient dicts, one per block
        """
        activations = [x]
        current = x

        for block in self.blocks:
            params = block.get_params()
            head_dim = self.embedding_dim // self.num_heads
            current = TransformerBlock.fwd(
                params, current, self.num_heads, head_dim, self.embedding_dim
            ) 
            activations.append(current)

        all_grads = []
        d_current = d_output

        for i in reversed(range(self.num_blocks)):
            block = self.blocks[i]
            block_input = activations[i]

            grads, d_current = block.compute_grads(block_input, d_current)
            all_grads.insert(0, grads)
        
        return all_grads
    
    def get_params_and_grads(self, all_grads=None):
        """
        Collect parameters and gradients from all blocks.

        Args:
            all_grads (list, optional): List of gradient dicts for each block

        Returns:
            list: Flattened list of all params and grads
        """
        if all_grads is None:
            all_grads = [None] * self.num_blocks
        
        result = []
        for block, grads in zip(self.blocks, all_grads):
            result.extend(block.get_params_and_grads(grads))
        
        return result
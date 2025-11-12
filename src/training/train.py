import os
# CRITICAL: Set this BEFORE importing JAX anywhere!
# Force JAX to use CPU (Apple Metal GPU support is buggy)
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import jax
import jax.numpy as jnp
import sys
from tqdm import tqdm
import time as t
import gc
import functools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pickle
from src.embeddings.embeddings import EmbeddingLayer
from src.transformer.transformer_stack import TransformerStack
from src.transformer.transformer_block import TransformerBlock
from src.transformer.output_layer import OutputLayer
from src.training.loss_function import CrossEntropyLoss
from src.tokenizer.tokenizer_class import BPETokenizer


class Trainer:
    """
    JAX-based trainer for transformer language model with stacked blocks.

    Key features:
    - Multiple stacked transformer blocks for deeper architecture
    - JAX autodiff for automatic gradient computation
    - JIT compilation for faster training
    - Multi-head attention (8 heads per block)

    Architecture:
        Embeddings → TransformerStack (4-6 blocks) → OutputLayer → Loss
    """

    def __init__(self, tokenizer, user_input, lr=1e-4, num_blocks=4, num_heads=8):
        """
        Initialize Trainer with model architecture.

        Args:
            tokenizer: BPE tokenizer instance
            user_input (list): List of text strings for training
            lr (float): Learning rate (default: 1e-4)
            num_blocks (int): Number of transformer blocks to stack (default: 4)
            num_heads (int): Number of attention heads per block (default: 8)
        """
        self.tokenizer = tokenizer
        self.token_ids = []

        for text in user_input:
            ids = tokenizer.encode(text)
            ids.append(tokenizer.eos_token_id)
            self.token_ids.append(ids)

        self.embedding_layer = EmbeddingLayer(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=512,
            max_seq_length=512  # Increased to handle longer sequences
        )

        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.transformer_stack = TransformerStack(
            self.embedding_layer,
            num_blocks=num_blocks,
            num_heads=num_heads
        )

        self.output_layer = OutputLayer(self.embedding_layer)
        self.loss_fn = CrossEntropyLoss()

        self.lr = lr

    def fwd(self, token_ids):
        """
        Forward pass through entire model using JAX.

        Args:
            token_ids (jnp.ndarray): Token IDs, shape (batch, seq_len)

        Returns:
            tuple: (transformer_out, logits)
        """
        embeddings, mask = EmbeddingLayer.embedding_fwd(
            (self.embedding_layer.embeddings, self.embedding_layer.positional_encodings),
            token_ids
        )
        transformer_out = self.transformer_stack.fwd(embeddings)

        output_params = self.output_layer.get_params()
        logits = OutputLayer.fwd(output_params, transformer_out)

        return transformer_out, logits

    def compute_loss_and_grads(self, token_ids, targets):
        """
        Compute loss and ALL gradients using JAX autodiff.

        Args:
            token_ids (jnp.ndarray): Input token IDs, shape (batch, seq_len)
            targets (jnp.ndarray): Target token IDs, shape (batch, seq_len)

        Returns:
            tuple: (loss, all_grads)
        """

        # Compute static architecture values outside the traced function
        num_heads = self.num_heads
        head_dim = self.embedding_layer.embedding_dim // self.num_heads
        embedding_dim = self.embedding_layer.embedding_dim

        def full_fwd_and_loss(embed_params, stack_params, output_params):
            embeddings, _ = EmbeddingLayer.embedding_fwd(embed_params, token_ids)

            current = embeddings
            for i, block in enumerate(self.transformer_stack.blocks):
                block_params = stack_params[i]

                current = TransformerBlock.fwd(
                    block_params,
                    current,
                    num_heads,
                    head_dim,
                    embedding_dim
                )

            logits = OutputLayer.fwd(output_params, current)
            loss = CrossEntropyLoss.fwd(logits, targets)

            return loss

        embed_params = (self.embedding_layer.embeddings, self.embedding_layer.positional_encodings)
        stack_params = [block.get_params() for block in self.transformer_stack.blocks]
        output_params = self.output_layer.get_params()

        loss, grads = jax.value_and_grad(
            full_fwd_and_loss,
            argnums=(0, 1, 2)
        )(embed_params, stack_params, output_params)

        embed_grads, stack_grads, output_grads = grads

        return loss, {
            'embeddings': embed_grads,
            'stack': stack_grads,
            'output': output_grads
        }

    def update_params(self, grads):
        """
        Update all parameters using computed gradients.

        Args:
            grads (dict): Gradients from compute_loss_and_grads()
        """
        # Update embedding layer
        embed_grads, pos_grads = grads['embeddings']
        self.embedding_layer.embeddings -= self.lr * embed_grads
        self.embedding_layer.positional_encodings -= self.lr * pos_grads

        # Update transformer stack (all blocks)
        for i, block in enumerate(self.transformer_stack.blocks):
            block_grads = grads['stack'][i]

            # Update attention params
            block.attention_layer.W_Q -= self.lr * block_grads['attn']['W_Q']
            block.attention_layer.W_K -= self.lr * block_grads['attn']['W_K']
            block.attention_layer.W_V -= self.lr * block_grads['attn']['W_V']
            block.attention_layer.W_O -= self.lr * block_grads['attn']['W_O']

            # Update FFN params
            block.ffn.W1 -= self.lr * block_grads['ffn']['W1']
            block.ffn.B1 -= self.lr * block_grads['ffn']['B1']
            block.ffn.W2 -= self.lr * block_grads['ffn']['W2']
            block.ffn.B2 -= self.lr * block_grads['ffn']['B2']

            # Update LayerNorm params
            block.gamma_1 -= self.lr * block_grads['gamma_1']
            block.beta_1 -= self.lr * block_grads['beta_1']
            block.gamma_2 -= self.lr * block_grads['gamma_2']
            block.beta_2 -= self.lr * block_grads['beta_2']

        # Update output layer
        self.output_layer.W_out -= self.lr * grads['output']['W_out']
        self.output_layer.b_out -= self.lr * grads['output']['b_out']

    def train(self, epochs=10, batch_size=20, checkpoint_path="artifacts/training_logs/training_logs.pkl", save_every=10):
        """
        Train the model with JAX autodiff.

        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            checkpoint_path (str): Path to save checkpoints
            save_every (int): Save checkpoint every N epochs
        """
        batches = self.create_batches(batch_size)

        gc.disable()

        try:
            for epoch in range(epochs):
                total_loss = 0

                for batch in tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                    start_time = t.time()

                    # Convert to JAX array
                    batch_jax = jnp.array(batch)

                    # Create targets (shift by 1 position)
                    input_tokens = batch_jax[:, :-1]
                    target_tokens = batch_jax[:, 1:]

                    # Forward + backward in ONE step (JAX magic!)
                    loss, grads = self.compute_loss_and_grads(input_tokens, target_tokens)

                    # Update parameters
                    self.update_params(grads)

                    total_loss += float(loss)  # Convert JAX scalar to Python float

                avg_loss = total_loss / len(batches)
                print(f"Epoch {epoch+1}/{epochs} complete. Avg loss: {avg_loss:.4f}")

                # Save checkpoint
                if (epoch + 1) % save_every == 0:
                    print(f"Saving checkpoint at epoch {epoch+1}...")
                    self.save_checkpoint(checkpoint_path)

                gc.collect()

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            print("Saving checkpoint before exit...")
            self.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            print(f"Training stopped at epoch {epoch+1}/{epochs}")
            gc.enable()
            raise

        gc.enable()
        print("Training complete! Saving final checkpoint...")
        self.save_checkpoint(checkpoint_path)

    def count_parameters(self):
        """
        Count total number of trainable parameters in the model.

        Returns:
            dict: Dictionary with parameter counts by component and total
        """
        param_counts = {
            'embedding': 0,
            'attention': 0,
            'feedforward': 0,
            'layer_norm': 0,
            'output': 0,
            'total': 0
        }

        # Embedding layer parameters
        param_counts['embedding'] += self.embedding_layer.embeddings.size
        param_counts['embedding'] += self.embedding_layer.positional_encodings.size

        # For each transformer block
        for block in self.transformer_stack.blocks:
            # Attention parameters
            param_counts['attention'] += block.attention_layer.W_Q.size
            param_counts['attention'] += block.attention_layer.W_K.size
            param_counts['attention'] += block.attention_layer.W_V.size
            param_counts['attention'] += block.attention_layer.W_O.size

            # Feedforward parameters
            param_counts['feedforward'] += block.ffn.W1.size
            param_counts['feedforward'] += block.ffn.B1.size
            param_counts['feedforward'] += block.ffn.W2.size
            param_counts['feedforward'] += block.ffn.B2.size

            # Layer normalization parameters
            param_counts['layer_norm'] += block.gamma_1.size
            param_counts['layer_norm'] += block.beta_1.size
            param_counts['layer_norm'] += block.gamma_2.size
            param_counts['layer_norm'] += block.beta_2.size

        # Output layer parameters
        param_counts['output'] += self.output_layer.W_out.size
        param_counts['output'] += self.output_layer.b_out.size

        # Total
        param_counts['total'] = sum(param_counts.values()) - param_counts['total']

        return param_counts

    def print_model_summary(self):
        """Print a summary of the model architecture and parameter counts."""
        counts = self.count_parameters()

        print("="*60)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*60)
        print(f"Vocabulary Size:      {self.tokenizer.vocab_size:,}")
        print(f"Embedding Dimension:  {self.embedding_layer.embedding_dim}")
        print(f"Max Sequence Length:  {self.embedding_layer.max_seq_length}")
        print(f"Number of Blocks:     {self.num_blocks}")
        print(f"Number of Heads:      {self.num_heads}")
        print(f"FFN Hidden Dimension: {self.transformer_stack.blocks[0].ffn.ff_dim}")
        print("="*60)
        print("PARAMETER COUNTS")
        print("="*60)
        print(f"Embedding Layer:      {counts['embedding']:>12,} parameters")
        print(f"Attention Layers:     {counts['attention']:>12,} parameters")
        print(f"FeedForward Layers:   {counts['feedforward']:>12,} parameters")
        print(f"Layer Normalization:  {counts['layer_norm']:>12,} parameters")
        print(f"Output Layer:         {counts['output']:>12,} parameters")
        print("-"*60)
        print(f"TOTAL:                {counts['total']:>12,} parameters")
        print("="*60)

        # Calculate model size in MB (assuming float32)
        size_mb = (counts['total'] * 4) / (1024 * 1024)
        print(f"Model Size (float32): ~{size_mb:.2f} MB")
        print("="*60)

    def save_checkpoint(self, path="artifacts/training_logs/training_logs.pkl"):
        """Save model parameters to file."""
        checkpoint = {
            'embeddings': self.embedding_layer.embeddings,
            'positional_encodings': self.embedding_layer.positional_encodings,
            'stack': [block.get_params() for block in self.transformer_stack.blocks],
            'output': self.output_layer.get_params(),
            'config': {
                'num_blocks': self.num_blocks,
                'num_heads': self.num_heads,
                'lr': self.lr,
                'vocab_size': self.tokenizer.vocab_size,
                'embedding_dim': self.embedding_layer.embedding_dim
            }
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path="artifacts/training_logs/training_logs.pkl"):
        """Load model parameters from file."""
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.embedding_layer.embeddings = checkpoint['embeddings']
        self.embedding_layer.positional_encodings = checkpoint['positional_encodings']

        for i, block_params in enumerate(checkpoint['stack']):
            block = self.transformer_stack.blocks[i]
            # Update attention
            block.attention_layer.W_Q = block_params['attn']['W_Q']
            block.attention_layer.W_K = block_params['attn']['W_K']
            block.attention_layer.W_V = block_params['attn']['W_V']
            block.attention_layer.W_O = block_params['attn']['W_O']
            # Update FFN
            block.ffn.W1 = block_params['ffn']['W1']
            block.ffn.B1 = block_params['ffn']['B1']
            block.ffn.W2 = block_params['ffn']['W2']
            block.ffn.B2 = block_params['ffn']['B2']
            # Update LayerNorm
            block.gamma_1 = block_params['gamma_1']
            block.beta_1 = block_params['beta_1']
            block.gamma_2 = block_params['gamma_2']
            block.beta_2 = block_params['beta_2']

        self.output_layer.W_out = checkpoint['output']['W_out']
        self.output_layer.b_out = checkpoint['output']['b_out']

        print(f"Loaded checkpoint from {path}")
        if 'config' in checkpoint:
            print(f"Config: {checkpoint['config']}")

    def generate(self, prompt, max_length=50, temperature=0.7, top_k=40, repetition_penalty=1.2):
        """
        Generate text using the trained model (JAX-based).

        Args:
            prompt (str): Input prompt
            max_length (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k filtering
            repetition_penalty (float): Penalty for repeating tokens

        Returns:
            str: Generated text
        """
        token_ids = self.tokenizer.encode(prompt)
        token_ids = [min(tid, self.tokenizer.vocab_size - 1) for tid in token_ids]

        for _ in range(max_length):
            # Convert to JAX array
            batch_token_ids = jnp.array([token_ids])

            # Forward pass
            transformer_out, logits = self.fwd(batch_token_ids)

            # Get logits for last token
            next_logits = jnp.array(logits[0, -1]) / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(token_ids):
                    if token_id < len(next_logits):
                        if next_logits[token_id] > 0:
                            next_logits = next_logits.at[token_id].set(
                                next_logits[token_id] / repetition_penalty
                            )
                        else:
                            next_logits = next_logits.at[token_id].set(
                                next_logits[token_id] * repetition_penalty
                            )

            # Top-k filtering
            if top_k is not None:
                top_k_indices = jnp.argsort(next_logits)[-top_k:]
                mask = jnp.ones_like(next_logits) * -jnp.inf
                mask = mask.at[top_k_indices].set(next_logits[top_k_indices])
                next_logits = mask

            # Sample
            probs = jax.nn.softmax(next_logits)
            probs = jnp.array(probs).flatten()

            # Convert to numpy for random choice (JAX doesn't have this)
            probs_np = np.array(probs)
            next_token = np.random.choice(len(probs_np), p=probs_np)

            next_token = min(int(next_token), self.tokenizer.vocab_size - 1)
            token_ids.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break

        # Decode
        try:
            return self.tokenizer.decode(token_ids)
        except (UnicodeDecodeError, Exception) as e:
            print(f"Warning: Decoding error: {e}")
            print(f"Generated token IDs: {token_ids[:20]}...")
            return "[Generation failed - invalid tokens produced]"

    def create_batches(self, batch_size=100):
        """Create padded batches."""
        batches = []
        for i in range(0, len(self.token_ids), batch_size):
            batch = self.token_ids[i:i+batch_size]
            max_len = max(len(seq) for seq in batch)
            padded_batches = [
                seq + [self.tokenizer.eos_token_id] * (max_len - len(seq))
                for seq in batch
            ]
            batches.append(np.array(padded_batches))
        return batches



def main():
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()


    user_input = "Hello world"
    trainer = Trainer(tokenizer, user_input, num_blocks=12, num_heads=12)

    trainer.print_model_summary()

if __name__ == '__main__':
    main()
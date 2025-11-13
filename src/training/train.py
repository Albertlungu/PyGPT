import os
import numpy as np
import jax
import jax.numpy as jnp
import sys
from tqdm import tqdm
import time as t
import gc
import functools
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pickle
from src.embeddings.embeddings import EmbeddingLayer
from src.transformer.transformer_stack import TransformerStack
from src.transformer.transformer_block import TransformerBlock
from src.transformer.output_layer import OutputLayer
from src.training.loss_function import CrossEntropyLoss
from src.tokenizer.tokenizer_class import BPETokenizer
from src.optimizers.adam import AdamNested


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
            embedding_dim=256,
            max_seq_length=256  # Increased to handle longer sequences
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

        # Initialize Adam optimizer
        self.optimizer = AdamNested(lr=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

        # Create JIT-compiled loss and gradient function
        self._compiled_loss_and_grad = self._create_jit_loss_fn()

    def _create_jit_loss_fn(self):
        """
        Create a JIT-compiled function for computing loss and gradients.
        This is created once during initialization for maximum performance.
        """
        num_heads = self.num_heads
        head_dim = self.embedding_layer.embedding_dim // self.num_heads
        embedding_dim = self.embedding_layer.embedding_dim
        num_blocks = self.num_blocks

        @jax.jit
        def loss_and_grad_fn(embed_params, stack_params, output_params, token_ids, targets):
            """JIT-compiled loss and gradient computation."""
            def loss_fn(embed_params, stack_params, output_params):
                embeddings, _ = EmbeddingLayer.embedding_fwd(embed_params, token_ids)

                current = embeddings
                for i in range(num_blocks):
                    block_params = stack_params[i]
                    current = TransformerBlock.fwd(
                        block_params,
                        current,
                        num_heads,
                        head_dim,
                        embedding_dim
                    )

                logits = OutputLayer.fwd(output_params, current)
                loss = CrossEntropyLoss.fwd(logits, targets, ignore_index=self.tokenizer.eos_token_id)
                return loss

            loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(
                embed_params, stack_params, output_params
            )
            return loss, grads

        return loss_and_grad_fn

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
        Compute loss and ALL gradients using JIT-compiled JAX autodiff.

        Args:
            token_ids (jnp.ndarray): Input token IDs, shape (batch, seq_len)
            targets (jnp.ndarray): Target token IDs, shape (batch, seq_len)

        Returns:
            tuple: (loss, all_grads)
        """
        embed_params = (self.embedding_layer.embeddings, self.embedding_layer.positional_encodings)
        stack_params = [block.get_params() for block in self.transformer_stack.blocks]
        output_params = self.output_layer.get_params()

        # Use the pre-compiled JIT function
        loss, grads = self._compiled_loss_and_grad(
            embed_params, stack_params, output_params, token_ids, targets
        )

        embed_grads, stack_grads, output_grads = grads

        return loss, {
            'embeddings': embed_grads,
            'stack': stack_grads,
            'output': output_grads
        }

    def update_params(self, grads):
        """
        Update all parameters using Adam optimizer.

        Args:
            grads (dict): Gradients from compute_loss_and_grads()
        """
        # Update embedding layer
        embed_grads, pos_grads = grads['embeddings']
        self.embedding_layer.embeddings = self.optimizer.step(
            self.embedding_layer.embeddings,
            embed_grads,
            path=('embeddings',)
        )
        self.embedding_layer.positional_encodings = self.optimizer.step(
            self.embedding_layer.positional_encodings,
            pos_grads,
            path=('positional',)
        )

        # Update transformer stack (all blocks)
        for i, block in enumerate(self.transformer_stack.blocks):
            block_grads = grads['stack'][i]

            # Update attention params
            block.attention_layer.W_Q = self.optimizer.step(
                block.attention_layer.W_Q,
                block_grads['attn']['W_Q'],
                path=('stack', i, 'attn', 'W_Q')
            )
            block.attention_layer.W_K = self.optimizer.step(
                block.attention_layer.W_K,
                block_grads['attn']['W_K'],
                path=('stack', i, 'attn', 'W_K')
            )
            block.attention_layer.W_V = self.optimizer.step(
                block.attention_layer.W_V,
                block_grads['attn']['W_V'],
                path=('stack', i, 'attn', 'W_V')
            )
            block.attention_layer.W_O = self.optimizer.step(
                block.attention_layer.W_O,
                block_grads['attn']['W_O'],
                path=('stack', i, 'attn', 'W_O')
            )

            # Update FFN params
            block.ffn.W1 = self.optimizer.step(
                block.ffn.W1,
                block_grads['ffn']['W1'],
                path=('stack', i, 'ffn', 'W1')
            )
            block.ffn.B1 = self.optimizer.step(
                block.ffn.B1,
                block_grads['ffn']['B1'],
                path=('stack', i, 'ffn', 'B1')
            )
            block.ffn.W2 = self.optimizer.step(
                block.ffn.W2,
                block_grads['ffn']['W2'],
                path=('stack', i, 'ffn', 'W2')
            )
            block.ffn.B2 = self.optimizer.step(
                block.ffn.B2,
                block_grads['ffn']['B2'],
                path=('stack', i, 'ffn', 'B2')
            )

            # Update LayerNorm params
            block.gamma_1 = self.optimizer.step(
                block.gamma_1,
                block_grads['gamma_1'],
                path=('stack', i, 'gamma_1')
            )
            block.beta_1 = self.optimizer.step(
                block.beta_1,
                block_grads['beta_1'],
                path=('stack', i, 'beta_1')
            )
            block.gamma_2 = self.optimizer.step(
                block.gamma_2,
                block_grads['gamma_2'],
                path=('stack', i, 'gamma_2')
            )
            block.beta_2 = self.optimizer.step(
                block.beta_2,
                block_grads['beta_2'],
                path=('stack', i, 'beta_2')
            )

        # Update output layer
        self.output_layer.W_out = self.optimizer.step(
            self.output_layer.W_out,
            grads['output']['W_out'],
            path=('output', 'W_out')
        )
        self.output_layer.b_out = self.optimizer.step(
            self.output_layer.b_out,
            grads['output']['b_out'],
            path=('output', 'b_out')
        )
        

    def _get_timestamped_checkpoint_path(self, base_path):
        """
        Generate a timestamped checkpoint path.

        Args:
            base_path (str): Base checkpoint path (e.g., "artifacts/training_logs/checkpoint.pkl")

        Returns:
            str: Timestamped path (e.g., "artifacts/training_logs/checkpoint_2025-01-11_14-30-45.pkl")
        """
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Split the path into directory, filename, and extension
        directory = os.path.dirname(base_path)
        filename = os.path.basename(base_path)
        name, ext = os.path.splitext(filename)

        # Create timestamped filename
        timestamped_filename = f"{name}_{timestamp}{ext}"
        timestamped_path = os.path.join(directory, timestamped_filename)

        return timestamped_path

    def train(self, epochs=10, batch_size=20, checkpoint_path="artifacts/training_logs/training_logs.pkl", save_every=10):
        """
        Train the model with JAX autodiff.
        Automatically saves checkpoints with timestamps.

        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            checkpoint_path (str): Base path for checkpoints (timestamp will be added)
            save_every (int): Save checkpoint every N epochs
        """
        # Generate timestamped checkpoint path at start of training
        timestamped_checkpoint = self._get_timestamped_checkpoint_path(checkpoint_path)
        print(f"Checkpoints will be saved to: {timestamped_checkpoint}")

        # print(f"Creating batches with batch_size={batch_size}...")
        batches = self.create_batches(batch_size)
        # print(f"Created {len(batches)} batches")

        gc.disable()

        try:
            for epoch in range(epochs):
                total_loss = 0
                print(f"\nStarting epoch {epoch+1}/{epochs}")

                for batch_idx, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}", leave=False, file=sys.stderr)):
                    start_time = t.time()

                    # print(f"[Epoch {epoch+1}, Batch {batch_idx+1}/{len(batches)}] Converting to JAX array...")
                    # Convert to JAX array
                    batch_jax = jnp.array(batch)
                    # print(f"  Shape: {batch_jax.shape}")

                    # Create targets (shift by 1 position)
                    input_tokens = batch_jax[:, :-1]
                    target_tokens = batch_jax[:, 1:]
                    # print(f"  Input shape: {input_tokens.shape}, Target shape: {target_tokens.shape}")

                    # print(f"  Computing loss and gradients (JIT compiling on first batch)...")
                    loss, grads = self.compute_loss_and_grads(input_tokens, target_tokens)
                    # print(f"  Loss computed: {float(loss):.4f}")

                    # Update parameters
                    # print(f"  Updating parameters...")
                    self.update_params(grads)

                    total_loss += float(loss)  # Convert JAX scalar to Python float

                    batch_time = t.time() - start_time
                    # print(f"  Batch complete in {batch_time:.2f}s")

                avg_loss = total_loss / len(batches)
                print(f"Epoch {epoch+1}/{epochs} complete. Avg loss: {avg_loss:.4f}")

                # Save checkpoint with timestamp
                if (epoch + 1) % save_every == 0:
                    print(f"Saving checkpoint at epoch {epoch+1}...")
                    self.save_checkpoint(timestamped_checkpoint)

                gc.collect()

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            print("Saving checkpoint before exit...")
            self.save_checkpoint(timestamped_checkpoint)
            print(f"Checkpoint saved to {timestamped_checkpoint}")
            print(f"Training stopped at epoch {epoch+1}/{epochs}")
            gc.enable()
            raise

        gc.enable()
        print("Training complete! Saving final checkpoint...")
        self.save_checkpoint(timestamped_checkpoint)

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
        size_mb = (counts['total'] * 2) / (1024 * 1024)
        print(f"Model Size (float16): ~{size_mb:.2f} MB")
        print("="*60)

    def save_checkpoint(self, path="artifacts/training_logs/training_logs.pkl"):
        """Save model parameters to file."""
        checkpoint = {
            'embeddings': self.embedding_layer.embeddings,
            'positional_encodings': self.embedding_layer.positional_encodings,
            'stack': [block.get_params() for block in self.transformer_stack.blocks],
            'output': self.output_layer.get_params(),
            'optimizer_state': self.optimizer.state,
            'optimizer_t': self.optimizer.t,
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

        # Restore optimizer state
        if 'optimizer_state' in checkpoint:
            self.optimizer.state = checkpoint['optimizer_state']
            self.optimizer.t = checkpoint['optimizer_t']

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
            str: Generated text (excluding the prompt)
        """
        token_ids = self.tokenizer.encode(prompt)
        token_ids = [min(tid, self.tokenizer.vocab_size - 1) for tid in token_ids]
        prompt_length = len(token_ids)  # Track original prompt length

        for _ in range(max_length):
            # Convert to JAX array
            batch_token_ids = jnp.array([token_ids])

            # Forward pass
            transformer_out, logits = self.fwd(batch_token_ids)

            # Get logits for last token
            next_logits = jnp.array(logits[0, -1]) / temperature

            # CRITICAL: Mask out invalid tokens (beyond vocab_size)
            # This ensures we NEVER sample invalid token IDs
            vocab_size = self.tokenizer.vocab_size
            if len(next_logits) > vocab_size:
                # Set logits for invalid tokens to -inf (probability = 0)
                next_logits = next_logits.at[vocab_size:].set(-jnp.inf)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(token_ids):
                    if token_id < vocab_size:
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

            # Normalize to ensure valid probability distribution
            probs_np = probs_np / probs_np.sum()

            next_token = np.random.choice(len(probs_np), p=probs_np)
            next_token = int(next_token)

            # This should never trigger now, but keep as final safety
            if next_token >= vocab_size:
                next_token = vocab_size - 1

            # Check for EOS BEFORE appending to avoid including it in output
            if next_token == self.tokenizer.eos_token_id:
                break

            token_ids.append(next_token)

        # Decode only the generated tokens (excluding the prompt)
        generated_token_ids = token_ids[prompt_length:]
        try:
            return self.tokenizer.decode(generated_token_ids), next_token, self.output_layer.b_out
        except (UnicodeDecodeError, Exception) as e:
            print(f"Warning: Decoding error: {e}")
            print(f"Generated token IDs: {generated_token_ids[:20]}...")
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
    
    def extend_training(self, checkpoint_path, epochs=10, batch_size=20, save_every=5):
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.load_checkpoint(checkpoint_path)

        base_name = "artifacts/training_logs/training_logs.pkl"
        new_checkpoint = self._get_timestamped_checkpoint_path(base_name)
        print(f"Extended training will save to {new_checkpoint}")

        print(f"Continuing training for {epochs} additional epochs...")
        self.train(
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_path=new_checkpoint,
            save_every=save_every
        )

        print(f"Extended training to {new_checkpoint}")




def main():
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()


    user_input = "Hello world"
    trainer = Trainer(tokenizer, user_input, num_blocks=12, num_heads=12)

    trainer.print_model_summary()

if __name__ == '__main__':
    main()
import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as tree
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

    def __init__(self, tokenizer, training_data=None, lr=1e-4, num_blocks=4, num_heads=8, embedding_dim=256, max_seq_length=256, use_lr_schedule=True, warmup_steps=500):
        """
        Initialize Trainer with model architecture.

        Args:
            tokenizer: BPE tokenizer instance
            user_input (list): List of text strings for training (default: None)
            pretokenized_data (loaded pkl file): The training data tokenized before training into a pkl file. Use pickle.load() to give here. (default: None)
            lr (float): Learning rate (default: 1e-4)
            num_blocks (int): Number of transformer blocks to stack (default: 4)
            num_heads (int): Number of attention heads per block (default: 8)
            embedding_dim (int): Embedding dimension (default: 256, must be divisible by num_heads)
            max_seq_length (int): Maximum sequence length for chunking (default: 256)
            use_lr_schedule (bool): Whether to use learning rate warmup and cosine decay (default: True)
            warmup_steps (int): Number of warmup steps (default: 500)
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.use_lr_schedule = use_lr_schedule
        self.warmup_steps = warmup_steps

        # Validate that embedding_dim is divisible by num_heads
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")

        self.embedding_layer = EmbeddingLayer(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length
        )

        self.token_ids = []

        # Process training data with proper chunking
        for text in training_data:
            ids = tokenizer.encode(text)

            # Chunk the sequence if it's too long
            if len(ids) > max_seq_length - 1:  # -1 to leave room for EOS
                # Split into chunks
                for i in range(0, len(ids), max_seq_length - 1):
                    chunk = ids[i:i + max_seq_length - 1]

                    # Only add EOS to the LAST chunk of this document
                    if i + max_seq_length - 1 >= len(ids):
                        chunk.append(tokenizer.eos_token_id)

                    self.token_ids.append(chunk)
            else:
                # Short sequence - just add EOS at the end
                ids.append(tokenizer.eos_token_id)
                self.token_ids.append(ids)


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

        # Initialize Adam optimizer (schedule will be set in train() when we know total steps)
        self.optimizer = AdamNested(lr=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

        # Create JIT-compiled loss and gradient function
        self._compiled_loss_and_grad = self._create_jit_loss_fn()

        # Create JIT-compiled update function
        self._compiled_update = self._create_jit_update_fn()

        # Initialize optimizer state (Adam moments) using pytree structure
        # This avoids initialization overhead on first batch
        initial_params = self._flatten_params()
        self._adam_m = tree.tree_map(lambda p: jnp.zeros_like(p), initial_params)
        self._adam_v = tree.tree_map(lambda p: jnp.zeros_like(p), initial_params)

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
                # Ignore padding (0) during loss calculation
                loss = CrossEntropyLoss.fwd(
                    logits,
                    targets,
                    ignore_indices=[self.tokenizer.padding_token_id],
                    eos_weight=1,  # Reduce EOS importance to prevent early stopping
                    eos_token_id=self.tokenizer.eos_token_id
                )
                return loss

            loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(
                embed_params, stack_params, output_params
            )
            return loss, grads

        return loss_and_grad_fn

    def _create_jit_update_fn(self):
        """
        Create a JIT-compiled function for updating parameters with Adam using pytrees.
        This is much faster than the tuple-based approach.
        """
        beta1 = self.optimizer.beta1
        beta2 = self.optimizer.beta2
        lr = self.optimizer.lr
        epsilon = self.optimizer.epsilon

        @jax.jit
        def update_fn(params_pytree, grads_pytree, optimizer_state, t):
            """
            JIT-compiled Adam update using pytrees (works on nested structures).

            Args:
                params_pytree: Nested dict of parameters (from _flatten_params)
                grads_pytree: Nested dict of gradients (same structure)
                optimizer_state: (m_pytree, v_pytree) - nested dicts of moment estimates
                t: Timestep

            Returns:
                (updated_params_pytree, new_optimizer_state)
            """
            m_pytree, v_pytree = optimizer_state

            def adam_update_leaf(param, grad, m, v):
                """Apply Adam update to a single parameter array."""
                # Update biased first moment
                m_new = beta1 * m + (1 - beta1) * grad

                # Update biased second moment
                v_new = beta2 * v + (1 - beta2) * (grad ** 2)

                # Bias correction
                m_hat = m_new / (1 - beta1 ** t)
                v_hat = v_new / (1 - beta2 ** t)

                # Update parameters
                param_new = param - lr * m_hat / (jnp.sqrt(v_hat) + epsilon)

                return param_new, m_new, v_new

            # Apply adam_update_leaf to every leaf in the pytree
            # This returns a pytree of tuples (param_new, m_new, v_new)
            result_pytree = tree.tree_map(
                adam_update_leaf,
                params_pytree,
                grads_pytree,
                m_pytree,
                v_pytree
            )

            # Unzip the tuples to get separate pytrees
            # JAX treats tuples as PyTree nodes, so we need to use tree_transpose
            # to properly unpack them
            from jax.tree_util import tree_transpose, tree_structure

            # Get the structure of the outer pytree (params) and inner structure (tuple of 3)
            outer_treedef = tree_structure(params_pytree)
            inner_treedef = tree_structure((0, 0, 0))  # 3-tuple structure

            # Transpose: outer structure of dicts/lists, inner structure of 3-tuples
            # -> inner structure of 3-tuples, outer structure of dicts/lists
            transposed = tree_transpose(outer_treedef, inner_treedef, result_pytree)

            # Now transposed is a 3-tuple of pytrees
            updated_params, updated_m, updated_v = transposed

            return updated_params, (updated_m, updated_v)

        return update_fn

    def _flatten_params(self):
        """
        Get all parameters as a pytree (tuple structure).
        This structure matches the gradient structure from compute_loss_and_grads.

        Returns:
            tuple: Nested tuple of all model parameters (embeddings, stack, output)
        """
        return (
            self.embedding_layer.get_params(),
            [block.get_params() for block in self.transformer_stack.blocks],
            self.output_layer.get_params()
        )

    def _unflatten_params(self, params):
        """
        Set all parameters from a pytree (tuple structure).

        Args:
            params (tuple): Nested tuple of parameters matching _flatten_params structure
                           (embeddings_dict, stack_list, output_dict)
        """
        embeddings_dict, stack_list, output_dict = params

        # Update embedding layer
        self.embedding_layer.embeddings = embeddings_dict['embeddings']
        self.embedding_layer.positional_encodings = embeddings_dict['positional_encodings']

        # Update transformer stack
        for i, block_params in enumerate(stack_list):
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

        # Update output layer
        self.output_layer.W_out = output_dict['W_out']
        self.output_layer.b_out = output_dict['b_out']

    def fwd(self, token_ids):
        """
        Forward pass through entire model using JAX.

        Args:
            token_ids (jnp.ndarray): Token IDs, shape (batch, seq_len)

        Returns:
            tuple: (transformer_out, logits)
        """
        embeddings, mask = EmbeddingLayer.embedding_fwd(
            self.embedding_layer.get_params(),
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
        embed_params = self.embedding_layer.get_params()
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
        Update all parameters using JIT-compiled Adam optimizer with pytrees.
        This is much faster than the previous list-based approach.

        Args:
            grads (dict): Gradients from compute_loss_and_grads()
                         Format: {'embeddings': dict, 'stack': list, 'output': dict}
        """
        # Get current parameters as pytree
        params_pytree = self._flatten_params()

        # Convert grads dict to tuple structure matching params_pytree
        grads_pytree = (
            grads['embeddings'],  # This is a dict {'embeddings': ..., 'positional_encodings': ...}
            grads['stack'],        # This is a list of dicts
            grads['output']        # This is a dict
        )

        # Increment timestep
        self.optimizer.t += 1

        # Run JIT-compiled update (all parameter updates happen in one JIT call)
        optimizer_state = (self._adam_m, self._adam_v)

        updated_params, new_state = self._compiled_update(
            params_pytree, grads_pytree, optimizer_state, self.optimizer.t
        )

        # Update optimizer state
        self._adam_m, self._adam_v = new_state

        # Unpack updated parameters back to model
        self._unflatten_params(updated_params)
        

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
                    batch_jax = jnp.array(batch, dtype=jnp.int32)
                    # print(batch_jax)
                    # print(f"    Batch #{batch_idx}")
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
        """Save model parameters AND optimizer state to file (for resuming training)."""
        checkpoint = {
            'embeddings': self.embedding_layer.embeddings,
            'positional_encodings': self.embedding_layer.positional_encodings,
            'stack': [block.get_params() for block in self.transformer_stack.blocks],
            'output': self.output_layer.get_params(),
            'optimizer_t': self.optimizer.t,
            'config': {
                'num_blocks': self.num_blocks,
                'num_heads': self.num_heads,
                'lr': self.lr,
                'vocab_size': self.tokenizer.vocab_size,
                'embedding_dim': self.embedding_layer.embedding_dim
            }
        }

        # Save optimizer state if it exists
        if hasattr(self, '_adam_m'):
            checkpoint['adam_m'] = self._adam_m
            checkpoint['adam_v'] = self._adam_v

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def save_model_only(self, path="artifacts/models/model.pkl"):
        """Save ONLY model weights (smaller file, for inference only)."""
        model_state = {
            'embeddings': self.embedding_layer.embeddings,
            'positional_encodings': self.embedding_layer.positional_encodings,
            'stack': [block.get_params() for block in self.transformer_stack.blocks],
            'output': self.output_layer.get_params(),
            'config': {
                'num_blocks': self.num_blocks,
                'num_heads': self.num_heads,
                'vocab_size': self.tokenizer.vocab_size,
                'embedding_dim': self.embedding_layer.embedding_dim
            }
        }

        with open(path, "wb") as f:
            pickle.dump(model_state, f)

        print(f"Model saved to {path} (weights only, no optimizer state)")

    def save_model_npz(self, path="artifacts/models/model.npz"):
        """Save model weights as compressed NumPy arrays (smallest file size)."""
        import numpy as np

        # Collect all parameters as numpy arrays
        save_dict = {
            'embeddings': np.array(self.embedding_layer.embeddings),
            'positional_encodings': np.array(self.embedding_layer.positional_encodings),
            'output_W': np.array(self.output_layer.W_out),
            'output_b': np.array(self.output_layer.b_out),
        }

        # Add transformer blocks
        for i, block in enumerate(self.transformer_stack.blocks):
            params = block.get_params()
            save_dict[f'block_{i}_W_Q'] = np.array(params['attn']['W_Q'])
            save_dict[f'block_{i}_W_K'] = np.array(params['attn']['W_K'])
            save_dict[f'block_{i}_W_V'] = np.array(params['attn']['W_V'])
            save_dict[f'block_{i}_W_O'] = np.array(params['attn']['W_O'])
            save_dict[f'block_{i}_W1'] = np.array(params['ffn']['W1'])
            save_dict[f'block_{i}_B1'] = np.array(params['ffn']['B1'])
            save_dict[f'block_{i}_W2'] = np.array(params['ffn']['W2'])
            save_dict[f'block_{i}_B2'] = np.array(params['ffn']['B2'])
            save_dict[f'block_{i}_gamma_1'] = np.array(params['gamma_1'])
            save_dict[f'block_{i}_beta_1'] = np.array(params['beta_1'])
            save_dict[f'block_{i}_gamma_2'] = np.array(params['gamma_2'])
            save_dict[f'block_{i}_beta_2'] = np.array(params['beta_2'])

        # Save config as metadata
        config = {
            'num_blocks': self.num_blocks,
            'num_heads': self.num_heads,
            'vocab_size': self.tokenizer.vocab_size,
            'embedding_dim': self.embedding_layer.embedding_dim
        }

        # Save with compression
        np.savez_compressed(path, **save_dict, config=config)
        print(f"Model saved to {path} (compressed NPZ format)")

    def load_checkpoint(self, path="artifacts/training_logs/training_logs.pkl"):
        """Load model parameters from file."""
        print(f"Loading checkpoint from {path}...")
        print("This may take 1-2 minutes for large files...")

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        print("Checkpoint loaded! Restoring model parameters...")

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
        if 'adam_m' in checkpoint:
            self._adam_m = checkpoint['adam_m']
            self._adam_v = checkpoint['adam_v']
            self.optimizer.t = checkpoint['optimizer_t']
            print("Loaded optimizer state from checkpoint")
        else:
            # Old checkpoint format or no optimizer state - reinitialize
            print("Warning: Old checkpoint format detected. Reinitializing optimizer state.")
            params_pytree = self._flatten_params()
            self._adam_m = tree.tree_map(lambda p: jnp.zeros_like(p), params_pytree)
            self._adam_v = tree.tree_map(lambda p: jnp.zeros_like(p), params_pytree)
            self.optimizer.t = 0

        print(f"Model parameters restored successfully!")
        if 'config' in checkpoint:
            print(f"Config: {checkpoint['config']}")
        print("Ready for inference!")

    def generate(self, prompt, max_length=50, temperature=0.7, top_k=40, repetition_penalty=1.2, debug=False):
        """
        Generate text using the trained model (JAX-based).

        Args:
            prompt (str): Input prompt
            max_length (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k filtering
            repetition_penalty (float): Penalty for repeating tokens
            debug (bool): Print debug information

        Returns:
            str: Generated text (excluding the prompt)
        """
        token_ids = self.tokenizer.encode(prompt)
        token_ids = [min(tid, self.tokenizer.vocab_size - 1) for tid in token_ids]
        prompt_length = len(token_ids)  # Track original prompt length

        if debug:
            print(f"\nDEBUG: Encoded prompt: {token_ids}")
            print(f"DEBUG: Prompt length: {prompt_length}")
            print(f"DEBUG: Vocab size: {self.tokenizer.vocab_size}")
            print(f"DEBUG: EOS token ID: {self.tokenizer.eos_token_id}")

        # Extract parameters once to avoid repeated dictionary lookups
        embed_params = self.embedding_layer.get_params()
        stack_params = [block.get_params() for block in self.transformer_stack.blocks]
        output_params = self.output_layer.get_params()

        # Force all operations to run on GPU if available
        with jax.default_device(jax.devices()[0]):
            for step_idx in range(max_length):
                # Convert to JAX array (JAX will automatically use GPU if available)
                batch_token_ids = jnp.array([token_ids], dtype=jnp.int32)

                # Forward pass
                transformer_out, logits = self.fwd(batch_token_ids)

                # Get logits for last token (keep as float32 for numerical stability)
                next_logits = logits[0, -1] / temperature

                # CRITICAL: Mask out invalid tokens (beyond vocab_size)
                # This ensures we NEVER sample invalid token IDs
                vocab_size = self.tokenizer.vocab_size

                if step_idx == 0 and debug:  # First token only
                    print(f"\nDEBUG First Generation Step:")
                    print(f"  Logits shape: {next_logits.shape}")
                    print(f"  Vocab size: {vocab_size}")
                    print(f"  Top 10 logit values: {jnp.sort(next_logits)[-10:]}")
                    print(f"  Top 10 token indices: {jnp.argsort(next_logits)[-10:]}")
                if len(next_logits) > vocab_size:
                    # Set logits for invalid tokens to -inf (probability = 0)
                    next_logits = next_logits.at[vocab_size:].set(-jnp.inf)

                # Repetition penalty (optimized - vectorized operation)
                if repetition_penalty != 1.0:
                    unique_tokens = jnp.array(list(set(token_ids)), dtype=jnp.int32)
                    # Filter out tokens >= vocab_size
                    unique_tokens = unique_tokens[unique_tokens < vocab_size]

                    # Apply penalty in a vectorized way
                    penalty_mask = jnp.zeros(vocab_size, dtype=jnp.bool_)
                    penalty_mask = penalty_mask.at[unique_tokens].set(True)

                    # Vectorized penalty application
                    penalties = jnp.where(
                        penalty_mask,
                        jnp.where(next_logits > 0, 1.0 / repetition_penalty, repetition_penalty),
                        1.0
                    )
                    next_logits = next_logits * penalties

                # Top-k filtering
                if top_k is not None:
                    top_k_indices = jnp.argsort(next_logits)[-top_k:]
                    mask = jnp.ones_like(next_logits) * -jnp.inf
                    mask = mask.at[top_k_indices].set(next_logits[top_k_indices])
                    next_logits = mask

                # Sample (optimized - keep in float32 for stability)
                probs = jax.nn.softmax(next_logits)
                probs_np = np.array(probs, dtype=np.float32)

                # Normalize to ensure valid probability distribution
                probs_np = probs_np / probs_np.sum()

                next_token = np.random.choice(len(probs_np), p=probs_np)
                next_token = int(next_token)

                if next_token >= vocab_size:
                    next_token = vocab_size - 1

                if debug:
                    print(f"DEBUG: Generated token {len(token_ids) - prompt_length + 1}: {next_token} (prob: {probs_np[next_token]:.4f})")
                    top_5_indices = np.argsort(probs_np)[-5:][::-1]
                    print(f"DEBUG: Top 5 tokens: {[(i, probs_np[i]) for i in top_5_indices]}")

                # Check for EOS BEFORE appending to avoid including it in output
                if next_token == self.tokenizer.eos_token_id:
                    if debug:
                        print(f"DEBUG: Hit EOS token at position {len(token_ids) - prompt_length}")
                    break

                token_ids.append(next_token)

        # Decode only the generated tokens (excluding the prompt)
        generated_token_ids = token_ids[prompt_length:]
        try:
            return self.tokenizer.decode(generated_token_ids), generated_token_ids
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
                seq + [0] * (max_len - len(seq))
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
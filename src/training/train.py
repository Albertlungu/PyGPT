import numpy as np
import sys, os
from tqdm import tqdm, trange
import time as t
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pickle
from src.embeddings.embeddings import EmbeddingLayer
from src.transformer.transformer_block import TransformerBlock
from src.transformer.output_layer import OutputLayer
from src.training.loss_function import CrossEntropyLoss


class Trainer:
    def __init__(self, tokenizer, user_input, lr = 1e-4):
        self.tokenizer = tokenizer
        self.token_ids = []

        for text in user_input:
            ids = tokenizer.encode(text)
            ids.append(tokenizer.eos_token_id)
            self.token_ids.append(ids)
        # print(self.tokenizer.decode(self.token_ids[1]))

        self.embedding_layer = EmbeddingLayer(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=256,
            max_seq_length=128  # Reduced from 512 to 128 (16x less memory for attention!)
        )
        # Initialize transformer without token_ids - we'll pass batches during forward
        # Pass empty list to satisfy constructor, but we'll override in fwd()
        self.transformer_block = TransformerBlock([], self.embedding_layer)
        self.output_layer = OutputLayer(self.embedding_layer)
        self.loss_fn = CrossEntropyLoss()

        self.lr = lr
        self.params = self.collect_params()
        self.transformer_out = None

    def fwd(self, token_ids):
        # Get embeddings for this batch
        embeddings = self.embedding_layer.fwd(token_ids)
        # Update transformer block's input embeddings for this batch
        self.transformer_block.input_embeddings = embeddings
        # Update attention and ffn layer batch/seq dimensions
        batch_size, seq_len, _ = embeddings.shape
        self.transformer_block.attention_layer.batch_size = batch_size
        self.transformer_block.attention_layer.seq_len = seq_len
        transformer_out = self.transformer_block.fwd()
        logits = self.output_layer.fwd(transformer_out)
        return transformer_out, logits
    
    def compute_loss(self, logits, targets):
        return self.loss_fn.fwd(logits, targets)
    
    def backward(self, logits, targets, transformer_out):
        probs = self.output_layer.softmax(logits)
        loss_grad = self.loss_fn.backward(logits, targets, probs)
        grad_to_transformer = self.output_layer.backward(loss_grad, transformer_out)
        self.transformer_block.backward(grad_to_transformer)

    def step(self):
        # Use cached parameter list instead of collecting every time
        for param in self.params:
            param['value'] -= self.lr * param['grad']

    def train(self, epochs = 10, batch_size = 20):  # Reduced from 100 to 20

        batches = self.create_batches(batch_size)

        # Disable automatic garbage collection during training
        # GC pauses cause significant delays between batches
        gc.disable()

        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(batches, desc = f"Epoch {epoch+1}/{epochs}", leave = False):
                start_time = t.time()
                self.zero_grad()
                time_zero_grad = t.time()-start_time


                start_fwd = t.time()
                # Get the actual sequence length from transformer output (may be truncated to max_seq_length)
                _, initial_seq_len, _ = self.embedding_layer.fwd(batch[:1]).shape  # Check what seq len we'll get

                # Truncate batch to match what embeddings will produce, then shift for targets
                truncated_batch = batch[:, :initial_seq_len]  # Shape: (batch_size, seq_len)
                targets = truncated_batch[:, 1:]  # Shape: (batch_size, seq_len-1)

                # Only pass the input portion (not the targets) to the model
                input_batch = truncated_batch[:, :-1]  # Shape: (batch_size, seq_len-1)

                transformer_out, logits = self.fwd(input_batch)
                time_fwd = t.time() - start_fwd


                start_loss = t.time()
                loss = self.compute_loss(logits, targets)
                time_loss = t.time() - start_loss


                start_bwd = t.time()
                # Use the outputs directly since they already match target shapes
                self.backward(logits, targets, transformer_out)
                time_bwd = t.time() - start_bwd

                start_step = t.time()
                self.clip_gradients()
                self.step()
                time_step = t.time() - start_step


                total_time = t.time() - start_time
                total_loss += loss

            print(f"ZeroGrad: {time_zero_grad:.4f}s, Fwd: {time_fwd:.4f}s, "
                f"Loss: {time_loss:.4f}s, Bwd: {time_bwd:.4f}s, Step: {time_step:.4f}s, "
                f"Total: {total_time:.4f}s")

            avg_loss = total_loss / len(batches)
            print(f"Epoch {epoch+1}/{epochs} complete. Avg loss: {avg_loss: .4f}")

            # Manually trigger garbage collection between epochs
            gc.collect()

        # Re-enable automatic garbage collection after training
        gc.enable()

    def collect_params(self):
        params = []

        params.extend(self.embedding_layer.get_params_and_grads())
        params.extend(self.transformer_block.get_params_and_grads())
        params.extend(self.output_layer.get_params_and_grads())

        return params

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
        embedding_params = self.embedding_layer.get_params_and_grads()
        for param in embedding_params:
            param_counts['embedding'] += param['value'].size

        # Attention layer parameters
        attention_params = self.transformer_block.attention_layer.get_params_and_grads()
        for param in attention_params:
            param_counts['attention'] += param['value'].size

        # Feedforward layer parameters
        ffn_params = self.transformer_block.ffn.get_params_and_grads()
        for param in ffn_params:
            param_counts['feedforward'] += param['value'].size

        # Layer normalization parameters (gamma and beta for both layer norms)
        param_counts['layer_norm'] += self.transformer_block.gamma_1.size
        param_counts['layer_norm'] += self.transformer_block.beta_1.size
        param_counts['layer_norm'] += self.transformer_block.gamma_2.size
        param_counts['layer_norm'] += self.transformer_block.beta_2.size

        # Output layer parameters
        output_params = self.output_layer.get_params_and_grads()
        for param in output_params:
            param_counts['output'] += param['value'].size

        # Total
        param_counts['total'] = sum(param_counts.values()) - param_counts['total']  # Subtract to avoid double counting

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
        print(f"FFN Hidden Dimension: {self.transformer_block.ffn.ff_dim}")
        print("="*60)
        print("PARAMETER COUNTS")
        print("="*60)
        print(f"Embedding Layer:      {counts['embedding']:>12,} parameters")
        print(f"Attention Layer:      {counts['attention']:>12,} parameters")
        print(f"FeedForward Layer:    {counts['feedforward']:>12,} parameters")
        print(f"Layer Normalization:  {counts['layer_norm']:>12,} parameters")
        print(f"Output Layer:         {counts['output']:>12,} parameters")
        print("-"*60)
        print(f"TOTAL:                {counts['total']:>12,} parameters")
        print("="*60)

        # Calculate model size in MB (assuming float32)
        size_mb = (counts['total'] * 4) / (1024 * 1024)
        print(f"Model Size (float32): ~{size_mb:.2f} MB")
        print("="*60)
    
    def zero_grad(self):
        # Use cached parameter list instead of collecting every time
        for param in self.params:
            param['grad'].fill(0)

    def clip_gradients(self, clip_value = 1.0):
        # Use cached parameter list instead of collecting every time
        for param in self.params:
            np.clip(param['grad'], -clip_value, clip_value, out = param['grad'])

    def save_checkpoint(self, path="artifacts/training_logs/training_logs.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.collect_params(), f)
    
    def load_checkpoint(self, path = "artifacts/training_logs/training_logs.pkl"):
        with open(path, "rb") as f:
            saved_params = pickle.load(f)
        for p, saved in zip(self.collect_params(), saved_params):
            p['value'][:] = saved['value']

    def generate(self, prompt, max_length = 50, temperature=1.0, top_k = None):
        token_ids = self.tokenizer.encode(prompt)

        # Clip token IDs to valid vocabulary range
        token_ids = [min(tid, self.tokenizer.vocab_size - 1) for tid in token_ids]

        for _ in range(max_length):
            # Need to pass as batch: shape (1, seq_len) instead of (seq_len,)
            batch_token_ids = np.array([token_ids])
            transformer_out, logits = self.fwd(batch_token_ids)

            # Get logits for last token in the sequence: shape (vocab_size,)
            next_logits = logits[0, -1] / temperature

            if top_k is not None:
                top_k_indices = np.argsort(next_logits)[-top_k:]
                mask = np.ones_like(next_logits) * -np.inf
                mask[top_k_indices] = next_logits[top_k_indices]
                next_logits = mask

            probs = np.exp(next_logits - np.max(next_logits))
            probs /= probs.sum()
            probs = np.array(probs).flatten()
            next_token = np.random.choice(len(probs), p = probs)

            # Clip to valid vocab range
            next_token = min(int(next_token), self.tokenizer.vocab_size - 1)

            token_ids.append(next_token)
            if next_token == self.tokenizer.eos_token_id:
                break


        # Validate and clean token IDs before decoding
        valid_ids = []
        for idx in token_ids:
            if idx < 0 or idx >= self.tokenizer.vocab_size:
                # Invalid ID, use EOS
                valid_ids.append(self.tokenizer.eos_token_id)
            elif idx in self.tokenizer.vocab:
                valid_ids.append(idx)
            else:
                # ID in range but not in vocab
                valid_ids.append(self.tokenizer.eos_token_id)

        try:
            return self.tokenizer.decode(valid_ids)
        except (UnicodeDecodeError, Exception) as e:
            print(f"Warning: Decoding error: {e}")
            print(f"Generated token IDs: {token_ids[:20]}...")  # Show first 20
            return "[Generation failed - invalid tokens produced]"
    
    def create_batches(self, batch_size = 100):
        batches = []
        for i in range(0, len(self.token_ids), batch_size):
            batch = self.token_ids[i:i+batch_size]
            max_len = max(len(seq) for seq in batch)
            padded_batches = [seq+[self.tokenizer.eos_token_id] * (max_len - len(seq)) for seq in batch]
            batches.append(np.array(padded_batches))
        return batches


# with open(training_path, 'r', encoding="utf-8") as f:
        #     text = f.read()
        # all_token_ids = self.tokenizer.encode(text)


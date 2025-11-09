import numpy as np
import sys, os
from tqdm import tqdm, trange
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

        self.embedding_layer = EmbeddingLayer(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=256,
            max_seq_length=512
        )
        self.transformer_block = TransformerBlock(self.token_ids, self.embedding_layer)
        self.output_layer = OutputLayer(self.embedding_layer)
        self.loss_fn = CrossEntropyLoss()

        self.lr = lr
        self.params = self.collect_params()
        self.transformer_out = None

    def fwd(self, token_ids):
        # embeddings = self.embedding_layer(token_ids)
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
        params = self.collect_params()
        for param in params:
            param['value'] -= self.lr * param['grad']

    def train(self, epochs=10, batch_size=16):
        for epoch in range(epochs):
            total_loss = 0
            batches = self.create_batches(batch_size)
            
            for batch_token_ids, lengths in batches:
                self.zero_grad()
                
                # Reinitialize transformer_block for this batch
                self.transformer_block = TransformerBlock(batch_token_ids, self.embedding_layer)
                
                transformer_out, logits = self.fwd(batch_token_ids)
                
                # Compute loss for each sequence in batch
                batch_loss = 0
                for b in range(len(batch_token_ids)):
                    seq_len = lengths[b]
                    targets = batch_token_ids[b][1:seq_len]
                    seq_logits = logits[b, :seq_len-1, :]
                    seq_transformer_out = transformer_out[b, :seq_len-1, :]
                    
                    loss = self.compute_loss(seq_logits, targets)
                    batch_loss += loss
                
                # Average loss over batch
                batch_loss /= len(batch_token_ids)
                
                # Backward pass with averaged gradients
                self.backward(logits[:, :-1, :], batch_token_ids, transformer_out[:, :-1, :])
                self.clip_gradients()
                self.step()
                
                total_loss += batch_loss
            
            avg_loss = total_loss / len(batches)
            print(f"Epoch {epoch+1}/{epochs} complete. Avg loss: {avg_loss:.4f}")


    def collect_params(self):
        params = []
        
        params.extend(self.embedding_layer.get_params_and_grads())
        params.extend(self.transformer_block.get_params_and_grads())
        params.extend(self.output_layer.get_params_and_grads())

        return params
    
    def zero_grad(self):
        for param in self.collect_params():
            param['grad'].fill(0)

    def clip_gradients(self, clip_value = 1.0):
        for param in self.collect_params():
            np.clip(param['grad'], -clip_value, clip_value, out = param['grad'])

    def save_checkpoint(self, path="artifacts/training_logs.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.collect_params(), f)
    
    def load_checkpoint(self, path = "artifacts/training_logs.pkl"):
        with open(path, "rb") as f:
            saved_params = pickle.load(f)
        for p, saved in zip(self.collect_params(), saved_params):
            p['value'][:] = saved['value']

    def generate(self, prompt, max_length = 50):
        token_ids = self.tokenizer.encode(prompt)
        for _ in range(max_length):
            transformer_out, logits = self.fwd(token_ids)
            next_token = np.argmax(logits[-1])
            # Ensure token is within valid vocab range
            if next_token >= self.tokenizer.vocab_size:
                break
            token_ids.append(int(next_token))
            if next_token == self.tokenizer.eos_token_id:
                break
        return self.tokenizer.decode(token_ids)
    

    def create_batches(self, batch_size=16):
        """
        Create batches of sequences with padding.
        
        Returns:
            List of batches, where each batch is (padded_token_ids, lengths)
        """
        import random
        random.shuffle(self.token_ids)
        
        batches = []
        for i in range(0, len(self.token_ids), batch_size):
            batch_sequences = self.token_ids[i:i + batch_size]
            
            # Find max length in this batch
            max_len = max(len(seq) for seq in batch_sequences)
            
            # Pad sequences to max_len
            padded_batch = []
            lengths = []
            for seq in batch_sequences:
                padded = seq + [0] * (max_len - len(seq))  # Pad with 0
                padded_batch.append(padded)
                lengths.append(len(seq))
            
            batches.append((padded_batch, lengths))
        
        return batches



# with open(training_path, 'r', encoding="utf-8") as f:
        #     text = f.read()
        # all_token_ids = self.tokenizer.encode(text)


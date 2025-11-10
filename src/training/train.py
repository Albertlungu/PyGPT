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

    def train(self, epochs = 10):
        for epoch in range(epochs):
            total_loss = 0
            for token_ids in tqdm(self.token_ids, desc = f"Epoch {epoch+1}/{epochs}", leave = False):
                self.zero_grad()
                transformer_out, logits = self.fwd(token_ids)
                targets = token_ids[1:]
                
                truncated_transformer_out = transformer_out[:-1]
                truncated_logits = logits[:-1]

                loss = self.compute_loss(truncated_logits, targets)

                self.backward(truncated_logits, targets, truncated_transformer_out)
                self.clip_gradients()
                self.step()

                total_loss += loss
            avg_loss = total_loss / len(self.token_ids)
            print(f"Epoch {epoch+1}/{epochs} complete. Avg loss: {avg_loss: .4f}")

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

    def save_checkpoint(self, path="artifacts/training_logs/training_logs.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.collect_params(), f)
    
    def load_checkpoint(self, path = "artifacts/training_logstraining_logs.pkl"):
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


# with open(training_path, 'r', encoding="utf-8") as f:
        #     text = f.read()
        # all_token_ids = self.tokenizer.encode(text)


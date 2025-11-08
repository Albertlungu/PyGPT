

# Building the Output Layer for Your LLM

This guide walks through creating a final output layer for your GPT-style model, using the existing components (`TransformerBlock`, `EmbeddingLayer`, `Attention`, `FeedForward`).

---

## 1. Purpose of the Output Layer

- [x] Convert the final transformer block embeddings into logits over the vocabulary.
- [x] For language modeling, each token in the sequence predicts the next token.
- [x] Output shape: `(batch_size, seq_len, vocab_size)`.
- [x] Optionally, tie the output layer weights with the token embeddings for efficiency.

---

## 2. Requirements

You already have:

- [x] `EmbeddingLayer` → for input token embeddings and positional encodings.
- [x] `TransformerBlock` → handles attention + feedforward + residuals.
- [x] `FeedForward` → used internally in blocks.
- [x] `Attention` → single-head attention for each block.
- [x] `BPETokenizer` → for encoding text into token IDs.

---

## 3. Implementation Steps

### Step 1: Create the OutputLayer class

- **File:** `output_layer.py`
- **Attributes:**
  - [x] `embedding_layer` → to access embedding dimension and possibly tie weights.
  - [x] `vocab_size` → number of tokens in tokenizer.
  - [x] `W_out` → weight matrix `(embedding_dim, vocab_size)`.
  - [x] `b_out` → bias vector `(vocab_size,)`.

```python
import numpy as np

class OutputLayer:
    def __init__(self, embedding_layer: EmbeddingLayer, vocab_size: int):
        self.embedding_layer = embedding_layer
        self.embedding_dim = embedding_layer.embedding_dim
        self.vocab_size = vocab_size

        # Weight tying option: use embedding matrix directly
        # self.W_out = embedding_layer.embeddings.T  # shape: (embedding_dim, vocab_size)
        self.W_out = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01
        self.b_out = np.zeros(self.vocab_size)
```

---

### Step 2: Forward Pass

- [x] Input: output of last `TransformerBlock` (`final_output`) → shape `(batch_size, seq_len, embedding_dim)`.
- [x] Compute logits:

```python
def fwd(self, transformer_output):
    """
    Args:
        transformer_output (np.ndarray): Output from last transformer block
                                         shape: (batch_size, seq_len, embedding_dim)
    Returns:
        np.ndarray: Logits over vocabulary, shape (batch_size, seq_len, vocab_size)
    """
    logits = transformer_output @ self.W_out + self.b_out
    return logits
```

- [x] Optional: apply softmax for probabilities:

```python
def softmax(self, logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probs
```

---

### Step 3: Integration with Transformer

```python
# Example forward pass
embedding_layer = EmbeddingLayer()
token_ids = tokenizer.encode("Hello World")
transformer_block = TransformerBlock(token_ids, embedding_layer)

# Forward through the block
block_output = transformer_block.fwd()  # shape: (batch, seq_len, embedding_dim)

# Forward through the output layer
output_layer = OutputLayer(embedding_layer, vocab_size=tokenizer.vocab_size)
logits = output_layer.fwd(block_output)
```

---

### Step 4: Training Considerations

- [ ] Loss function: use cross-entropy loss between predicted logits and true next token IDs.
- [ ] Ensure gradients flow back to:
  - [ ] Output layer weights (`W_out`, `b_out`)
  - [ ] Transformer block weights
  - [ ] Embedding layer (if not weight-tied)

---

### Step 5: Optional Features

- [ ] Weight tying: set `W_out = embedding_layer.embeddings.T` to share weights and save parameters.
- [ ] Masked prediction: if doing autoregressive LM, ensure attention masks prevent tokens from seeing future tokens.
- [ ] Batch support: forward pass should handle `(batch_size, seq_len)` sequences.

---

### Checklist Before Implementation

- [ ] `TransformerBlock` outputs `(batch_size, seq_len, embedding_dim)`
- [ ] `EmbeddingLayer` dimension matches transformer output (`embedding_dim`)
- [ ] Tokenizer provides `vocab_size`
- [ ] Decide if you want weight tying or separate output weights
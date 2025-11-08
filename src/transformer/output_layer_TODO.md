# Output Layer Class - Build Guide

## Overview
The Output Layer is the final component of your GPT-style language model. It converts the rich contextual embeddings from the last TransformerBlock into probability distributions over your vocabulary, enabling the model to predict the next token in a sequence.

**This is where your model's understanding becomes words!**

---

## Prerequisites

Before building the Output Layer, you must have:
- [x] Completed and tested EmbeddingLayer class
- [x] Completed and tested TransformerBlock class
- [x] Working tokenizer with defined vocab_size

---

## Input Specification

### What comes in:
- **Shape**: `(batch_size, sequence_length, embedding_dimension)`
- **Type**: Processed embeddings from the final TransformerBlock
- **From where**: Output of the last TransformerBlock in your model stack
- **NOT**: Raw embeddings or token IDs

### Variable name explanations:
- **batch_size**: How many sequences you're processing at once (e.g., 32)
- **sequence_length**: Number of tokens in each sequence (e.g., 128)
- **embedding_dimension**: Size of each token's vector representation (e.g., 512)

### What it represents:
Rich contextual embeddings that encode information about each token's meaning in context, ready to be decoded into vocabulary predictions.

---

## Parameters to Initialize

### Checklist - Initialization:

- [ ] **embedding_layer**: Reference to your EmbeddingLayer instance
  - [ ] Pass the EmbeddingLayer object
  - [ ] **What it does**: Provides access to embedding_dim and vocab_size
  - [ ] Enables weight tying (optional optimization)

- [ ] **embedding_dim**: Dimension of input embeddings
  - [ ] Extracted from: `embedding_layer.embedding_dim`
  - [ ] Typical values: 256, 512, 768, 1024
  - [ ] **What it does**: Defines input size for the output projection

- [ ] **vocab_size**: Size of the vocabulary
  - [ ] Extracted from: `embedding_layer.vocab_size`
  - [ ] Same size as your tokenizer's vocabulary
  - [ ] **What it does**: Defines output size (one logit per vocabulary token)

- [ ] **W_out**: Output projection weight matrix
  - [ ] Shape: `(embedding_dim, vocab_size)`
  - [ ] **Option 1 - Weight Tying**: `W_out = embedding_layer.embeddings.T`
  - [ ] **Option 2 - Separate Weights**: `W_out = np.random.randn(embedding_dim, vocab_size) * 0.01`
  - [ ] **What it does**: Projects embeddings to vocabulary space

- [ ] **b_out**: Output bias vector
  - [ ] Shape: `(vocab_size,)`
  - [ ] Initialize: `np.zeros(vocab_size)`
  - [ ] **What it does**: Adds learnable bias to each vocabulary token's logit

### Understanding Weight Tying:

**Weight Tying** is a technique where the output layer shares weights with the embedding layer:

**Advantages**:
- ✅ Reduces total parameters (more efficient)
- ✅ Creates symmetry: similar embeddings → similar predictions
- ✅ Often improves generalization

**Implementation**:
```python
# Weight tying: use transposed embedding matrix
self.W_out = embedding_layer.embeddings.T  # shape: (embedding_dim, vocab_size)
```

**Separate Weights**:
```python
# Independent output weights
self.W_out = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01
```

**When to use which**:
- Use weight tying: Default choice for most language models (GPT-style)
- Use separate weights: When you need more flexibility or embeddings are frozen

---

## Forward Pass Implementation

### Architecture Overview:
```
Transformer Output (batch, seq_len, embedding_dim)
    ↓
[Linear Projection: @ W_out + b_out]
    ↓
Logits (batch, seq_len, vocab_size)
    ↓
[Softmax] (optional, for inference)
    ↓
Probabilities (batch, seq_len, vocab_size)
```

---

### Step 1: Linear Projection to Vocabulary Space

- [ ] Multiply transformer output with W_out
  - [ ] Operation: `logits = transformer_output @ W_out + b_out`
  - [ ] Input shape: `(batch, seq_len, embedding_dim)`
  - [ ] W_out shape: `(embedding_dim, vocab_size)`
  - [ ] b_out shape: `(vocab_size,)` - broadcasts across batch and sequence
  - [ ] Output shape: `(batch, seq_len, vocab_size)`

**What's happening**: Each token's embedding (size embedding_dim) is projected to a vector of size vocab_size, where each element represents the unnormalized score (logit) for that vocabulary token.

**Variable names**:
- **logits**: Unnormalized scores for each vocabulary token
- Each logits[b, s, v] represents the model's raw score for token v at position s in batch b

---

### Step 2: Softmax (For Inference/Prediction)

**Note**: Softmax is typically NOT applied during training (loss function handles this). Only use for generating predictions.

- [ ] Implement numerically stable softmax
- [ ] Input: logits of shape `(batch, seq_len, vocab_size)`
- [ ] Output: probabilities of shape `(batch, seq_len, vocab_size)`

**Implementation Checklist**:

- [ ] **Subtract max for numerical stability**:
  - [ ] `max_logits = np.max(logits, axis=-1, keepdims=True)`
  - [ ] `shifted_logits = logits - max_logits`
  - [ ] **Why**: Prevents overflow from exp() of large numbers

- [ ] **Compute exponentials**:
  - [ ] `exp_logits = np.exp(shifted_logits)`
  - [ ] Shape stays: `(batch, seq_len, vocab_size)`

- [ ] **Normalize to probabilities**:
  - [ ] `sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)`
  - [ ] `probs = exp_logits / sum_exp`
  - [ ] Verify: Each probability vector sums to 1.0

**What's happening**: Converting raw scores (logits) into valid probability distributions. Each position gets a distribution over the entire vocabulary.

**Variable names**:
- **probs**: Probability distribution over vocabulary for each token position
- probs[b, s, :] sums to 1.0 and represents P(next_token | context) at position s

---

## Output Specification

### What comes out (Forward Pass):
- **Shape**: `(batch_size, sequence_length, vocab_size)`
- **Type**: Logits (unnormalized scores) or probabilities (if softmax applied)
- **Goes to**:
  - Training: Loss function (cross-entropy)
  - Inference: Sampling/greedy selection for next token prediction

### Understanding the Output:

For language modeling, we typically care about predicting the **next** token:
- Position 0 predicts token at position 1
- Position 1 predicts token at position 2
- Position i predicts token at position i+1

---

## Helper Functions to Implement

### 1. Forward Pass (Required)

```python
def fwd(self, transformer_output):
    """
    Projects transformer output to vocabulary logits.

    Args:
        transformer_output (np.ndarray): Output from last TransformerBlock
                                         shape: (batch_size, seq_len, embedding_dim)

    Returns:
        np.ndarray: Logits over vocabulary
                    shape: (batch_size, seq_len, vocab_size)
    """
```

**Checklist**:
- [ ] Validate input shape has 3 dimensions
- [ ] Verify input shape[-1] == self.embedding_dim
- [ ] Compute logits: `transformer_output @ self.W_out + self.b_out`
- [ ] Verify output shape: `(batch, seq_len, vocab_size)`
- [ ] Return logits

---

### 2. Softmax Function (For Inference)

```python
def softmax(self, logits):
    """
    Converts logits to probability distributions.

    Args:
        logits (np.ndarray): Unnormalized scores
                             shape: (batch_size, seq_len, vocab_size)

    Returns:
        np.ndarray: Probability distributions
                    shape: (batch_size, seq_len, vocab_size)
    """
```

**Checklist**:
- [ ] Subtract max along last dimension for stability
- [ ] Compute exponentials
- [ ] Sum along last dimension
- [ ] Divide to get probabilities
- [ ] Verify each position sums to ~1.0

---

### 3. Predict Next Token (For Generation)

```python
def predict_next_token(self, transformer_output, temperature=1.0):
    """
    Samples the next token from the model's predictions.

    Args:
        transformer_output (np.ndarray): Output from last TransformerBlock
                                         shape: (batch_size, seq_len, embedding_dim)
        temperature (float): Sampling temperature (default 1.0)
                            Higher = more random, Lower = more deterministic

    Returns:
        np.ndarray: Predicted token IDs
                    shape: (batch_size,) - one prediction per sequence
    """
```

**Implementation Checklist**:

- [ ] **Get logits for last position**:
  - [ ] `logits = self.fwd(transformer_output)[:, -1, :]`  # shape: (batch, vocab_size)
  - [ ] **Why**: Last position predicts next token

- [ ] **Apply temperature scaling**:
  - [ ] `scaled_logits = logits / temperature`
  - [ ] temperature < 1.0 → more confident (peaked distribution)
  - [ ] temperature > 1.0 → more random (flatter distribution)

- [ ] **Convert to probabilities**:
  - [ ] `probs = self.softmax(scaled_logits)`

- [ ] **Sample from distribution**:
  - [ ] For each sequence in batch, sample token according to probabilities
  - [ ] **Greedy**: `predicted_tokens = np.argmax(probs, axis=-1)`
  - [ ] **Sampling**: Use np.random.choice with probabilities

- [ ] Return predicted token IDs

**Variable names**:
- **temperature**: Controls randomness (1.0 = normal, <1.0 = deterministic, >1.0 = creative)
- **predicted_tokens**: Integer token IDs selected from vocabulary

---

## Complete Forward Pass Flow

```
transformer_output: (batch, seq_len, embedding_dim)
    ↓
logits = transformer_output @ W_out + b_out
    ↓
logits: (batch, seq_len, vocab_size)
    ↓
[During Training]
    → logits passed to loss function

[During Inference]
    ↓
probs = softmax(logits / temperature)
    ↓
predicted_token = sample(probs) or argmax(probs)
    ↓
return predicted_token
```

---

## Integration with Full Model

### Example: Building Complete Forward Pass

```python
# 1. Tokenize input
token_ids = tokenizer.encode("Hello world")

# 2. Get embeddings with positional encoding
embedding_layer = EmbeddingLayer(vocab_size=tokenizer.vocab_size)
embeddings = embedding_layer.fwd(token_ids)

# 3. Pass through transformer blocks (could be multiple)
transformer_block = TransformerBlock(embedding_dim=256, num_heads=8, ff_dim=1024)
transformer_output = transformer_block.fwd(embeddings)

# 4. Project to vocabulary
output_layer = OutputLayer(embedding_layer)
logits = output_layer.fwd(transformer_output)

# 5. Generate next token
next_token = output_layer.predict_next_token(transformer_output, temperature=0.8)
```

---

## Build Strategy: Step-by-Step Approach

### Implementation Steps:

- [ ] **Step 1**: Initialize the OutputLayer class
  - [ ] Store embedding_layer reference
  - [ ] Get embedding_dim and vocab_size
  - [ ] Decide: weight tying or separate weights
  - [ ] Initialize W_out and b_out

- [ ] **Step 2**: Implement forward pass (fwd method)
  - [ ] Take transformer_output as input
  - [ ] Compute logits via linear projection
  - [ ] Verify shapes

- [ ] **Step 3**: Implement softmax helper function
  - [ ] Numerically stable implementation
  - [ ] Apply along correct axis

- [ ] **Step 4**: Test with simple inputs
  - [ ] Create dummy transformer output
  - [ ] Run forward pass
  - [ ] Check logits shape and values

- [ ] **Step 5**: Implement predict_next_token (optional, for generation)
  - [ ] Extract last position logits
  - [ ] Apply temperature
  - [ ] Sample or select greedily

- [ ] **Step 6**: Test with real transformer output
  - [ ] Connect to actual TransformerBlock
  - [ ] Verify end-to-end forward pass

---

## Testing Checklist

### Basic Functionality Tests:

- [ ] Create test transformer output
  - [ ] batch_size = 2
  - [ ] sequence_length = 10
  - [ ] embedding_dimension = 256

- [ ] Initialize OutputLayer with:
  - [ ] embedding_layer with vocab_size = 5000
  - [ ] embedding_dimension = 256

- [ ] Run forward pass

- [ ] Verify output shape: Should be `(2, 10, 5000)`

### Value Tests:

- [ ] Check for errors:
  - [ ] No NaN values in logits
  - [ ] No Inf values in logits

- [ ] Softmax check:
  - [ ] All probabilities between 0 and 1
  - [ ] Each position sums to approximately 1.0

- [ ] Prediction check:
  - [ ] Predicted token IDs are integers
  - [ ] Token IDs are in valid range: 0 to vocab_size-1

### Integration Tests:

- [ ] Test with actual TransformerBlock output
- [ ] Test with different batch sizes
- [ ] Test temperature scaling (0.5, 1.0, 2.0)
- [ ] Verify weight tying if used (W_out should reference embeddings)

---

## Common Issues & Debugging

### Issue: Shape mismatch in matrix multiplication
- [ ] Verify transformer_output shape is `(batch, seq_len, embedding_dim)`
- [ ] Check W_out shape is `(embedding_dim, vocab_size)`
- [ ] Print shapes at each step

### Issue: Softmax returns NaN
- [ ] Ensure you subtract max before exp()
- [ ] Check for Inf values in logits before softmax
- [ ] Verify axis=-1 for softmax normalization

### Issue: All predictions are the same token
- [ ] Check if W_out initialized properly (not all zeros)
- [ ] Verify b_out is reasonable (not huge values)
- [ ] Ensure transformer is providing meaningful outputs

### Issue: Weight tying not working
- [ ] Verify W_out is a reference: `W_out = embeddings.T` (not a copy)
- [ ] Check embedding_layer.embeddings shape is `(vocab_size, embedding_dim)`
- [ ] After transpose, W_out should be `(embedding_dim, vocab_size)`

---

## Understanding Language Model Predictions

### How Next-Token Prediction Works:

Given input sequence: "The cat sat on the"

1. **Tokenize**: `[5, 42, 198, 67, 8]`
2. **Embed**: Convert to embeddings
3. **Transform**: Pass through TransformerBlocks
4. **Project**: OutputLayer creates logits
5. **Predict**:
   - Position 0 logits → used to predict token at position 1
   - Position 4 logits → used to predict the NEXT token (position 5)

**For generation**: We care about the **last position**'s logits to predict what comes next.

---

## Backward Pass (For Training)

### Checklist - Backward Implementation:

- [ ] **Input**: Gradient from loss function
  - [ ] Shape: `(batch, seq_len, vocab_size)`

- [ ] **Compute W_out gradient**:
  - [ ] `dW_out = transformer_output.T @ gradient_from_loss`
  - [ ] Need to reshape/sum appropriately for batches
  - [ ] Shape: `(embedding_dim, vocab_size)`

- [ ] **Compute b_out gradient**:
  - [ ] `db_out = np.sum(gradient_from_loss, axis=(0, 1))`
  - [ ] Sum over batch and sequence dimensions
  - [ ] Shape: `(vocab_size,)`

- [ ] **Compute gradient w.r.t. transformer_output**:
  - [ ] `d_transformer_output = gradient_from_loss @ W_out.T`
  - [ ] Shape: `(batch, seq_len, embedding_dim)`
  - [ ] Pass this back to TransformerBlock

- [ ] **Update weights** (if using optimizer):
  - [ ] `W_out -= learning_rate * dW_out`
  - [ ] `b_out -= learning_rate * db_out`

**Note on Weight Tying**: If using weight tying, gradient for W_out also updates the embedding layer!

---

## Advanced Features (Optional)

### 1. Top-k Sampling

- [ ] Instead of sampling from full distribution, sample from top k most likely tokens
- [ ] Reduces chance of selecting very unlikely tokens

### 2. Top-p (Nucleus) Sampling

- [ ] Sample from smallest set of tokens whose cumulative probability exceeds p
- [ ] More dynamic than top-k

### 3. Beam Search

- [ ] Instead of greedy/sampling, explore multiple likely sequences
- [ ] Useful for tasks requiring best single output (translation, summarization)

---

## Example Dimensions

For a typical small GPT-style model:
- embedding_dimension = 512
- vocab_size = 10,000
- batch_size = 32
- sequence_length = 128

**Complete Flow**:
- Transformer output: `(32, 128, 512)`
- W_out: `(512, 10000)`
- Logits: `(32, 128, 10000)`
- After softmax: `(32, 128, 10000)` where each `[:,:,:]` are probabilities
- Next token prediction: `(32,)` - one token per sequence

---

## Glossary of Terms

- **logits**: Unnormalized scores for each vocabulary token
- **softmax**: Function that converts logits to probability distributions
- **temperature**: Parameter controlling randomness in sampling (1.0 = normal)
- **weight tying**: Sharing weights between embedding and output layers
- **greedy decoding**: Always selecting most probable token (argmax)
- **sampling**: Randomly selecting token according to probability distribution
- **cross-entropy loss**: Training loss that compares predicted and true token distributions
- **vocab_size**: Total number of unique tokens in your vocabulary

---

## Success Criteria

You've successfully built the Output Layer when:

- [ ] Output shape is `(batch, seq_len, vocab_size)` for any valid input
- [ ] Softmax produces valid probability distributions (sum to 1)
- [ ] No NaN or Inf values in outputs
- [ ] Can predict next tokens from transformer output
- [ ] Temperature scaling works correctly
- [ ] Weight tying properly references embeddings (if used)
- [ ] Ready to connect to loss function for training!

---

## Next Steps

After completing the Output Layer:
1. Implement loss function (cross-entropy) for training
2. Build training loop to update all model parameters
3. Test generation with temperature and sampling strategies
4. Evaluate model performance on held-out text

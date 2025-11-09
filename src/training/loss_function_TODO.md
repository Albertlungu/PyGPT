# Loss Function Class - Build Guide

## Overview
The Loss Function is the critical component that tells your model how wrong its predictions are. For language models, we use **Cross-Entropy Loss** to compare the model's predicted token probabilities against the actual next tokens in the training data.

**This is how your model learns what's right and what's wrong!**

---

## Prerequisites

Before building the Loss Function, you should understand:
- [x] How the OutputLayer produces logits
- [x] What softmax does (converts logits to probabilities)
- [x] Basic concept of gradients and backpropagation
- [x] Shape: `(batch_size, sequence_length, vocab_size)` for logits

---

## What is Cross-Entropy Loss?

### Intuitive Explanation:

Cross-entropy measures the difference between two probability distributions:
1. **Model's prediction**: Probability distribution over vocab from your OutputLayer
2. **True distribution**: The actual next token (one-hot encoded)

**Lower loss** = Better predictions = Model assigns high probability to correct tokens

**Higher loss** = Worse predictions = Model assigns low probability to correct tokens

### Example:

Given vocabulary: `["the", "cat", "dog", "sat"]` (indices 0, 1, 2, 3)

**True next token**: "cat" (index 1)

**Model predictions** (after softmax):
- [0.1, 0.7, 0.1, 0.1] → **Low loss** (model predicts "cat" with 70% confidence) ✅
- [0.4, 0.2, 0.3, 0.1] → **High loss** (model only gives "cat" 20% probability) ❌

---

## Input Specification

### What comes in:

#### 1. Logits (from OutputLayer)
- **Shape**: `(batch_size, sequence_length, vocab_size)`
- **Type**: Unnormalized scores (floats, can be negative)
- **From where**: OutputLayer.fwd() output
- **What it is**: Raw predictions before softmax

#### 2. Target Token IDs (ground truth)
- **Shape**: `(batch_size, sequence_length)`
- **Type**: Integer token IDs
- **From where**: Training data (the actual next tokens)
- **What it is**: Correct answers for the model to learn

### Variable name explanations:
- **batch_size**: How many sequences you're training on at once (e.g., 32)
- **sequence_length**: Number of tokens in each sequence (e.g., 128)
- **vocab_size**: Total number of unique tokens (e.g., 10,000)

### Important Alignment:

For next-token prediction:
- Logits at position `i` predict the token at position `i+1`
- Target at position `i` should be the token ID at position `i+1` from original sequence

---

## Cross-Entropy Loss Formula

### Mathematical Definition:

For a single prediction:

$$\text{Loss} = -\log(p_{\text{true}})$$

Where:
- $p_{\text{true}}$ = Probability assigned to the true token
- $\log$ = Natural logarithm

### For Multiple Predictions:

$$\text{Total Loss} = -\frac{1}{N} \sum_{i=1}^{N} \log(p_{\text{true}, i})$$

Where:
- $N$ = Total number of predictions (batch_size × sequence_length)

### Why Negative Log?

- When $p_{\text{true}} = 1.0$ (perfect prediction): $-\log(1.0) = 0$ (no loss)
- When $p_{\text{true}} = 0.5$ (uncertain): $-\log(0.5) \approx 0.69$
- When $p_{\text{true}} = 0.1$ (bad prediction): $-\log(0.1) \approx 2.30$ (high loss)

The log function heavily penalizes low probabilities!

---

## Implementation Strategy

### Two Approaches:

**Approach 1: Numerically Stable (Recommended)**
- Compute loss directly from logits using log-softmax
- More stable, prevents overflow/underflow
- Standard in modern frameworks

**Approach 2: Explicit Softmax**
- First compute softmax, then compute log
- Easier to understand but less stable
- Good for learning

We'll cover **both** approaches below.

---

## Parameters to Initialize

### Checklist - Initialization:

The CrossEntropyLoss class typically doesn't need parameters to initialize, but you may want:

- [x] **ignore_index**: Token ID to ignore in loss computation (optional)
  - [x] Default: -1 or None
  - [x] Use case: Ignore padding tokens
  - [x] **What it does**: Doesn't compute loss for positions with this token ID

- [x] **reduction**: How to aggregate loss values (optional)
  - [x] Options: "mean" (average over all positions) or "sum"
  - [x] Default: "mean"
  - [x] **What it does**: Controls output: single number vs per-sample loss

---

## Forward Pass Implementation (Approach 1: Stable)

### Step 1: Extract Shape Information

- [x] Get batch_size, seq_len, vocab_size from logits shape
  - [x] `batch_size, seq_len, vocab_size = logits.shape`

- [x] Verify targets shape matches `(batch_size, seq_len)`
  - [x] Targets should be integers in range [0, vocab_size-1]

**Variable names**:
- **logits**: Predictions from model, shape `(batch, seq_len, vocab_size)`
- **targets**: True token IDs, shape `(batch, seq_len)`

---

### Step 2: Compute Log-Softmax (Numerically Stable)

Instead of computing softmax then taking log, combine them for stability:

$$\text{log\_softmax}(x_i) = x_i - \log(\sum_j \exp(x_j))$$

**Implementation Checklist**:

- [x] **Subtract max for stability**:
  - [x] `max_logits = np.max(logits, axis=-1, keepdims=True)`
  - [x] `shifted_logits = logits - max_logits`
  - [x] Shape: `(batch, seq_len, vocab_size)`

- [x] **Compute log-sum-exp**:
  - [x] `exp_logits = np.exp(shifted_logits)`
  - [x] `sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)`
  - [x] `log_sum_exp = np.log(sum_exp)`
  - [x] Shape of log_sum_exp: `(batch, seq_len, 1)`

- [x] **Compute log probabilities**:
  - [x] `log_probs = shifted_logits - log_sum_exp`
  - [x] Shape: `(batch, seq_len, vocab_size)`
  - [x] Each log_probs[b, s, :] represents log probabilities at position s

**What's happening**: Computing the log of probabilities without actually computing probabilities (more stable!).

**Variable names**:
- **log_probs**: Log probabilities for each token, shape `(batch, seq_len, vocab_size)`

---

### Step 3: Select Log Probabilities of True Tokens

We need to extract the log probability assigned to each true target token.

**Implementation Checklist**:

- [ ] **Flatten logits for easier indexing**:
  - [ ] `log_probs_flat = log_probs.reshape(-1, vocab_size)`
  - [ ] Shape: `(batch * seq_len, vocab_size)`

- [ ] **Flatten targets**:
  - [ ] `targets_flat = targets.reshape(-1)`
  - [ ] Shape: `(batch * seq_len,)`

- [ ] **Extract log probabilities of true tokens**:
  - [ ] `selected_log_probs = log_probs_flat[np.arange(len(targets_flat)), targets_flat]`
  - [ ] Shape: `(batch * seq_len,)`
  - [ ] **What this does**: For each position, select the log prob of the true token

**Explanation of indexing**:
```python
# For each position i, we want log_probs[i, targets[i]]
# np.arange(len(targets_flat)) creates [0, 1, 2, ..., n-1]
# targets_flat contains the column indices (which token was true)
# This fancy indexing extracts the right log probability for each position
```

**Variable names**:
- **selected_log_probs**: Log probabilities assigned to true tokens only

---

### Step 4: Compute Negative Log Likelihood

- [ ] Negate the selected log probabilities:
  - [ ] `negative_log_likelihood = -selected_log_probs`
  - [ ] Shape: `(batch * seq_len,)`

**What's happening**: Converting log probabilities to loss values. Higher probability → lower loss.

**Variable names**:
- **negative_log_likelihood**: Loss value for each prediction

---

### Step 5: Handle Padding (Optional)

If you have padding tokens in your sequences (e.g., token ID = 0):

- [ ] **Create mask for valid positions**:
  - [ ] `valid_mask = (targets_flat != ignore_index)`
  - [ ] Shape: `(batch * seq_len,)` of booleans

- [ ] **Apply mask**:
  - [ ] `masked_loss = negative_log_likelihood * valid_mask`
  - [ ] Positions with padding get 0 loss

- [ ] **Adjust averaging**:
  - [ ] `num_valid = np.sum(valid_mask)`
  - [ ] Use this instead of total elements when computing mean

**Variable names**:
- **valid_mask**: Boolean array indicating which positions are not padding
- **num_valid**: Count of non-padding positions

---

### Step 6: Aggregate Loss

- [ ] **Compute mean loss**:
  - [ ] If no padding: `loss = np.mean(negative_log_likelihood)`
  - [ ] If padding: `loss = np.sum(masked_loss) / num_valid`
  - [ ] Output: Single scalar value

- [ ] **Alternative - Per-sample loss**:
  - [ ] `per_sample_loss = negative_log_likelihood.reshape(batch_size, seq_len)`
  - [ ] `loss = np.mean(per_sample_loss, axis=1)`  # Average over sequence
  - [ ] Output: `(batch_size,)` - one loss per sequence

**What's happening**: Combining all individual losses into a single number to minimize during training.

**Variable names**:
- **loss**: Final scalar loss value (or per-sample if desired)

---

## Forward Pass Implementation (Approach 2: Explicit)

This approach is easier to understand but less numerically stable.

### Step-by-Step:

- [ ] **Step 1**: Compute softmax from logits
  - [ ] `max_logits = np.max(logits, axis=-1, keepdims=True)`
  - [ ] `exp_logits = np.exp(logits - max_logits)`
  - [ ] `probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)`

- [ ] **Step 2**: Add small epsilon to avoid log(0)
  - [ ] `epsilon = 1e-10`
  - [ ] `probs_safe = np.clip(probs, epsilon, 1.0)`

- [ ] **Step 3**: Compute log of probabilities
  - [ ] `log_probs = np.log(probs_safe)`

- [ ] **Step 4**: Select log probs of true tokens (same as Approach 1)

- [ ] **Step 5**: Compute mean negative log likelihood

---

## Backward Pass (Gradient Computation)

The gradient of cross-entropy loss w.r.t. logits has a beautiful simple form!

### Mathematical Result:

$$\frac{\partial \text{Loss}}{\partial \text{logits}} = \text{probs} - \text{one\_hot\_targets}$$

Where:
- **probs**: Softmax probabilities from logits
- **one_hot_targets**: One-hot encoded version of target token IDs

### Implementation Checklist:

- [ ] **Compute softmax probabilities**:
  - [ ] If not already computed: `probs = softmax(logits)`
  - [ ] Shape: `(batch, seq_len, vocab_size)`

- [ ] **Create one-hot encoding of targets**:
  - [ ] `one_hot_targets = np.zeros_like(logits)`  # All zeros initially
  - [ ] Shape: `(batch, seq_len, vocab_size)`

- [ ] **Set true token positions to 1**:
  - [ ] Flatten for easier indexing
  - [ ] `one_hot_flat = one_hot_targets.reshape(-1, vocab_size)`
  - [ ] `targets_flat = targets.reshape(-1)`
  - [ ] `one_hot_flat[np.arange(len(targets_flat)), targets_flat] = 1`
  - [ ] `one_hot_targets = one_hot_flat.reshape(batch, seq_len, vocab_size)`

- [ ] **Compute gradient**:
  - [ ] `d_logits = probs - one_hot_targets`
  - [ ] Shape: `(batch, seq_len, vocab_size)`

- [ ] **Scale by batch size** (for mean reduction):
  - [ ] `d_logits = d_logits / (batch_size * seq_len)`
  - [ ] Or if ignoring padding: `d_logits = d_logits / num_valid`

- [ ] **Return gradient**:
  - [ ] `return d_logits`
  - [ ] This gradient flows back to OutputLayer

**What's happening**:
- Where model predicted correct token: gradient pushes probability higher
- Where model predicted wrong tokens: gradient pushes probabilities lower
- The difference between prediction and truth is the gradient!

**Variable names**:
- **d_logits**: Gradient of loss with respect to logits
- **one_hot_targets**: Binary array with 1 at true token position, 0 elsewhere

---

## Complete Loss Class Structure

```python
class CrossEntropyLoss:
    def __init__(self, ignore_index=None, reduction='mean'):
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.probs = None  # Store for backward pass
        self.targets = None  # Store for backward pass

    def forward(self, logits, targets):
        """
        Compute cross-entropy loss.

        Args:
            logits: (batch, seq_len, vocab_size)
            targets: (batch, seq_len)

        Returns:
            loss: scalar
        """
        # Implementation here
        pass

    def backward(self):
        """
        Compute gradient of loss w.r.t. logits.

        Returns:
            d_logits: (batch, seq_len, vocab_size)
        """
        # Implementation here
        pass
```

---

## Output Specification

### Forward Pass Output:
- **Shape**: Scalar (single number) or `(batch_size,)` if per-sample
- **Type**: Float
- **Range**: 0 to infinity (lower is better)
- **Typical values**:
  - Random model: ~log(vocab_size) ≈ 9.2 for vocab_size=10,000
  - Trained model: 2-5 depending on task

### Backward Pass Output:
- **Shape**: `(batch_size, sequence_length, vocab_size)`
- **Type**: Float gradients
- **Goes to**: OutputLayer for backpropagation

---

## Testing Checklist

### Basic Functionality Tests:

- [ ] **Test 1: Perfect predictions**
  - [ ] Create logits where true token has very high score
  - [ ] Targets match the highest scoring tokens
  - [ ] Expected loss: Close to 0

- [ ] **Test 2: Random predictions**
  - [ ] Create uniform random logits
  - [ ] Expected loss: Close to log(vocab_size)

- [ ] **Test 3: Shape verification**
  - [ ] Input: logits `(2, 5, 100)`, targets `(2, 5)`
  - [ ] Output: scalar loss
  - [ ] Gradient: shape `(2, 5, 100)`

### Value Tests:

- [ ] **Loss is non-negative**
  - [ ] All loss values ≥ 0

- [ ] **Loss decreases with better predictions**
  - [ ] Modify logits to favor true tokens
  - [ ] Verify loss goes down

- [ ] **Gradient sums to zero** (approximately)
  - [ ] For each position, sum of gradients across vocab should ≈ 0
  - [ ] This is a property of softmax gradient

### Numerical Stability Tests:

- [ ] **Test with very large logits** (e.g., 1000)
  - [ ] Should not produce NaN or Inf

- [ ] **Test with very small logits** (e.g., -1000)
  - [ ] Should not produce NaN or Inf

- [ ] **Test with very large vocab_size** (e.g., 50,000)
  - [ ] Should handle memory efficiently

---

## Common Issues & Debugging

### Issue: Loss is NaN
- [ ] Check for log(0) - ensure epsilon added to probabilities
- [ ] Check for Inf in logits - may need gradient clipping
- [ ] Verify max subtraction in softmax

### Issue: Loss is not decreasing during training
- [ ] Verify targets are correct (not shifted by 1)
- [ ] Check learning rate (too small or too large)
- [ ] Ensure gradients are flowing back correctly

### Issue: Loss is negative
- [ ] This should never happen! Check implementation
- [ ] Verify you're using negative log likelihood

### Issue: Gradient shape mismatch
- [ ] Ensure gradient shape matches logits shape exactly
- [ ] Check one-hot encoding creates correct shape

---

## Understanding Loss Values

### What different loss values mean:

- **Loss ≈ log(vocab_size)**: Random guessing
  - For vocab_size = 10,000: ~9.2
  - For vocab_size = 50,000: ~10.8

- **Loss = 5-7**: Partially trained model
  - Model has learned some patterns
  - Still room for improvement

- **Loss = 2-4**: Well-trained model
  - Model understands language structure
  - Makes reasonable predictions

- **Loss < 2**: Very good model (or overfitting)
  - Highly accurate predictions
  - Check if overfitting on training data

- **Loss = 0**: Perfect predictions (or bug)
  - Only achievable with 100% accuracy
  - On real data, this suggests a bug

---

## Integration with Training Loop

### Example Usage:

```python
# Initialize
loss_fn = CrossEntropyLoss()
output_layer = OutputLayer(embedding_layer)

# Forward pass
embeddings = embedding_layer.fwd(token_ids)
transformer_output = transformer_block.fwd(embeddings)
logits = output_layer.fwd(transformer_output)

# Compute loss
targets = token_ids[:, 1:]  # Shift by 1 for next-token prediction
logits_for_loss = logits[:, :-1, :]  # Remove last position
loss = loss_fn.forward(logits_for_loss, targets)

# Backward pass
d_logits = loss_fn.backward()
# Continue backprop through output_layer, transformer_block, etc.
```

---

## Advanced Topics (Optional)

### 1. Label Smoothing

Instead of one-hot targets, smooth the distribution:
- True token: probability = 1 - ε (e.g., 0.9)
- All other tokens: probability = ε / (vocab_size - 1)

**Benefits**: Prevents overconfidence, improves generalization

---

### 2. Sequence-Level Weighting

Weight different positions in sequence differently:
- Give more weight to later tokens (harder to predict)
- Give less weight to common tokens (like "the")

---

### 3. Perplexity Metric

Convert loss to perplexity for interpretability:
- Perplexity = exp(loss)
- Perplexity ≈ "effective vocab size" the model is confused over
- Lower perplexity = better model

**Example**:
- Loss = 3.0 → Perplexity ≈ 20 (model confused over ~20 tokens)
- Loss = 6.0 → Perplexity ≈ 403 (model very confused)

---

## Glossary of Terms

- **cross-entropy**: Measure of difference between two probability distributions
- **log-likelihood**: Log of probability assigned to true data
- **negative log-likelihood**: Loss function = -log-likelihood
- **one-hot encoding**: Binary vector with 1 at true class, 0 elsewhere
- **softmax**: Function converting logits to probability distribution
- **log-softmax**: Numerically stable combination of log and softmax
- **perplexity**: Exponential of loss, measures model uncertainty

---

## Success Criteria

You've successfully built the Loss Function when:

- [ ] Computes scalar loss from logits and targets
- [ ] Loss is always non-negative
- [ ] Perfect predictions give loss ≈ 0
- [ ] Random predictions give loss ≈ log(vocab_size)
- [ ] No NaN or Inf values
- [ ] Gradient shape matches logits shape
- [ ] Gradient flows back to OutputLayer correctly
- [ ] Ready to plug into training loop!

---

## Next Steps

After completing the Loss Function:
1. Implement training loop to update model parameters
2. Monitor loss during training (should decrease)
3. Compute validation loss to check for overfitting
4. Calculate perplexity for easier interpretation
5. Test model generation after training

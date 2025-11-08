# Training Loop - Build Guide

## Overview
The Training Loop is where all your model components come together and actually learn! This is the heart of the training process, orchestrating forward passes, loss computation, backpropagation, and parameter updates across many iterations.

**This is where your model transforms from random weights to a language generator!**

---

## Prerequisites

Before building the Training Loop, you must have:
- [x] Completed EmbeddingLayer class with forward/backward methods
- [x] Completed TransformerBlock class with forward/backward methods
- [x] Completed OutputLayer class with forward/backward methods
- [x] Completed CrossEntropyLoss class
- [x] Working tokenizer (BPETokenizer)
- [x] Training data prepared as text files

---

## What is a Training Loop?

### The Big Picture:

Training a language model involves:
1. **Feed data through model** (forward pass)
2. **Compute how wrong the model is** (loss)
3. **Calculate how to improve** (gradients via backpropagation)
4. **Update model parameters** (optimization)
5. **Repeat thousands/millions of times** (epochs)

### One Training Iteration:

```
Input tokens → Embeddings → TransformerBlock(s) → OutputLayer → Logits
                                                                    ↓
                                                              Compute Loss
                                                                    ↓
                                                              Backpropagate
                                                                    ↓
                                                           Update Parameters
```

---

## Training Data Preparation

### What You Need:

- [ ] **Text corpus**: Large text file(s) for training
  - [ ] Should be representative of target domain
  - [ ] Minimum: ~1MB of text (more is better)
  - [ ] Already tokenized using your BPETokenizer

- [ ] **Sequence batching**: Convert text to fixed-length sequences
  - [ ] Typical sequence length: 128, 256, or 512 tokens
  - [ ] Batch size: 8, 16, 32, or 64 sequences
  - [ ] Each batch: `(batch_size, seq_len)` of token IDs

---

## Input Specification

### Training Data Format:

- **Input sequences**: `(batch_size, sequence_length)` of token IDs
- **Target sequences**: Same as input, but shifted by 1 position
  - If input[i] = [5, 42, 198, 67, 8]
  - Then target[i] = [42, 198, 67, 8, <next_token>]

### Why shifted?
For language modeling, we predict the **next** token:
- Input token at position 0 → predicts token at position 1
- Input token at position 1 → predicts token at position 2
- And so on...

---

## Hyperparameters to Define

### Checklist - Hyperparameters:

- [ ] **learning_rate**: How big the parameter updates are
  - [ ] Typical values: 1e-4, 3e-4, 1e-3
  - [ ] Too high → unstable training
  - [ ] Too low → very slow learning
  - [ ] **Recommended start**: 3e-4

- [ ] **batch_size**: Number of sequences per iteration
  - [ ] Typical values: 8, 16, 32, 64
  - [ ] Larger batch = more stable gradients, more memory
  - [ ] Smaller batch = less memory, noisier gradients
  - [ ] **Recommended start**: 16

- [ ] **sequence_length**: Max tokens per sequence
  - [ ] Typical values: 128, 256, 512
  - [ ] Longer sequences = more context, more memory
  - [ ] **Recommended start**: 128

- [ ] **num_epochs**: How many times to loop through entire dataset
  - [ ] Typical values: 3-10 for large datasets
  - [ ] More epochs = more training, risk of overfitting
  - [ ] **Recommended start**: 5

- [ ] **max_iterations**: Alternative to epochs (total update steps)
  - [ ] Typical values: 10,000 - 1,000,000
  - [ ] Use this OR num_epochs, not both

- [ ] **gradient_clip_value**: Maximum allowed gradient magnitude
  - [ ] Prevents exploding gradients
  - [ ] Typical values: 1.0, 5.0
  - [ ] **Recommended**: 1.0

- [ ] **eval_interval**: How often to evaluate on validation set
  - [ ] Every N iterations
  - [ ] Typical values: 100, 500, 1000
  - [ ] **Recommended**: 500

- [ ] **save_interval**: How often to save model checkpoints
  - [ ] Every N iterations
  - [ ] Typical values: 1000, 5000
  - [ ] **Recommended**: 1000

---

## Model Architecture Setup

### Components to Initialize:

- [ ] **Tokenizer**:
  ```python
  with open('artifacts/tokenizer.pkl', 'rb') as f:
      tokenizer = pickle.load(f)
  ```

- [ ] **EmbeddingLayer**:
  ```python
  embedding_layer = EmbeddingLayer(
      vocab_size=tokenizer.vocab_size,
      embedding_dim=256,
      max_seq_length=512
  )
  ```

- [ ] **TransformerBlock(s)**:
  ```python
  # Single block (simplest)
  transformer_block = TransformerBlock(
      embedding_dim=256,
      num_heads=8,  # if using multi-head
      ff_dim=1024   # typically 4 × embedding_dim
  )

  # Or multiple blocks (more powerful)
  num_layers = 6
  transformer_blocks = [
      TransformerBlock(256, 8, 1024)
      for _ in range(num_layers)
  ]
  ```

- [ ] **OutputLayer**:
  ```python
  output_layer = OutputLayer(embedding_layer)
  ```

- [ ] **Loss Function**:
  ```python
  loss_fn = CrossEntropyLoss()
  ```

---

## Data Loading Implementation

### Step 1: Load Raw Text

- [ ] Read text file(s):
  ```python
  with open('data/training_text.txt', 'r', encoding='utf-8') as f:
      text = f.read()
  ```

- [ ] Verify text loaded correctly:
  - [ ] Check length: `len(text)`
  - [ ] Preview first 100 characters

---

### Step 2: Tokenize Text

- [ ] Encode entire text to token IDs:
  ```python
  all_token_ids = tokenizer.encode(text)
  ```

- [ ] Verify tokenization:
  - [ ] Check total tokens: `len(all_token_ids)`
  - [ ] Decode sample to verify: `tokenizer.decode(all_token_ids[:100])`

---

### Step 3: Create Sequences

- [ ] Split into fixed-length sequences:
  ```python
  def create_sequences(token_ids, seq_len):
      sequences = []
      for i in range(0, len(token_ids) - seq_len, seq_len):
          seq = token_ids[i : i + seq_len]
          sequences.append(seq)
      return np.array(sequences)

  sequences = create_sequences(all_token_ids, sequence_length)
  ```

- [ ] Shape verification:
  - [ ] `sequences.shape = (num_sequences, sequence_length)`

---

### Step 4: Create Batches

- [ ] Batch sequences together:
  ```python
  def create_batches(sequences, batch_size):
      num_batches = len(sequences) // batch_size
      # Trim to multiple of batch_size
      sequences = sequences[:num_batches * batch_size]
      # Reshape into batches
      batches = sequences.reshape(num_batches, batch_size, -1)
      return batches

  batches = create_batches(sequences, batch_size)
  ```

- [ ] Shape verification:
  - [ ] `batches.shape = (num_batches, batch_size, sequence_length)`

---

### Step 5: Train/Validation Split

- [ ] Split data for evaluation:
  ```python
  split_idx = int(0.9 * len(batches))  # 90% train, 10% validation
  train_batches = batches[:split_idx]
  val_batches = batches[split_idx:]
  ```

---

## Training Loop Implementation

### Main Training Loop Structure:

```python
def train(model_components, train_batches, val_batches, hyperparams):
    """
    Main training loop.

    Args:
        model_components: Dict with embedding_layer, transformer_blocks,
                         output_layer, loss_fn
        train_batches: Training data batches
        val_batches: Validation data batches
        hyperparams: Dict with learning_rate, num_epochs, etc.
    """
```

---

### Step 1: Setup Training

- [ ] Extract hyperparameters:
  ```python
  learning_rate = hyperparams['learning_rate']
  num_epochs = hyperparams['num_epochs']
  gradient_clip = hyperparams['gradient_clip_value']
  eval_interval = hyperparams['eval_interval']
  save_interval = hyperparams['save_interval']
  ```

- [ ] Initialize tracking variables:
  ```python
  iteration = 0
  train_losses = []
  val_losses = []
  ```

---

### Step 2: Epoch Loop

- [ ] Loop over epochs:
  ```python
  for epoch in range(num_epochs):
      print(f"Epoch {epoch + 1}/{num_epochs}")

      # Shuffle batches each epoch
      np.random.shuffle(train_batches)
  ```

---

### Step 3: Batch Loop (Inner Loop)

- [ ] Loop over batches:
  ```python
  for batch_idx, batch in enumerate(train_batches):
      iteration += 1

      # Extract input and target
      input_ids = batch[:, :-1]    # All but last token
      target_ids = batch[:, 1:]    # All but first token
  ```

**Important**: Input and target are shifted by 1 for next-token prediction!

**Variable names**:
- **input_ids**: Tokens to feed as input, shape `(batch_size, seq_len - 1)`
- **target_ids**: Tokens to predict, shape `(batch_size, seq_len - 1)`

---

### Step 4: Forward Pass

- [ ] **Pass through embedding layer**:
  ```python
  embeddings = embedding_layer.fwd(input_ids)
  # Shape: (batch_size, seq_len - 1, embedding_dim)
  ```

- [ ] **Pass through transformer block(s)**:
  ```python
  # Single block
  transformer_output = transformer_block.fwd(embeddings)

  # Multiple blocks
  hidden = embeddings
  for block in transformer_blocks:
      hidden = block.fwd(hidden)
  transformer_output = hidden
  # Shape: (batch_size, seq_len - 1, embedding_dim)
  ```

- [ ] **Pass through output layer**:
  ```python
  logits = output_layer.fwd(transformer_output)
  # Shape: (batch_size, seq_len - 1, vocab_size)
  ```

**Variable names**:
- **embeddings**: Token embeddings with positional encoding
- **transformer_output**: Contextualized representations
- **logits**: Unnormalized scores for each vocabulary token

---

### Step 5: Compute Loss

- [ ] Calculate loss:
  ```python
  loss = loss_fn.forward(logits, target_ids)
  # Returns: scalar value
  ```

- [ ] Track loss:
  ```python
  train_losses.append(loss)
  ```

---

### Step 6: Backward Pass

- [ ] **Compute gradient from loss**:
  ```python
  d_logits = loss_fn.backward()
  # Shape: (batch_size, seq_len - 1, vocab_size)
  ```

- [ ] **Backprop through output layer**:
  ```python
  d_transformer_output = output_layer.backward(d_logits)
  # Shape: (batch_size, seq_len - 1, embedding_dim)
  ```

- [ ] **Backprop through transformer block(s)**:
  ```python
  # Single block
  d_embeddings = transformer_block.backward(d_transformer_output)

  # Multiple blocks (reverse order!)
  d_hidden = d_transformer_output
  for block in reversed(transformer_blocks):
      d_hidden = block.backward(d_hidden)
  d_embeddings = d_hidden
  # Shape: (batch_size, seq_len - 1, embedding_dim)
  ```

- [ ] **Backprop through embedding layer**:
  ```python
  embedding_layer.backward(d_embeddings)
  # Accumulates gradients internally
  ```

---

### Step 7: Gradient Clipping

Prevent exploding gradients:

- [ ] **Clip gradients** (for each component):
  ```python
  def clip_gradients(gradients, max_value):
      """Clip gradients to max_value."""
      grad_norm = np.linalg.norm(gradients)
      if grad_norm > max_value:
          gradients = gradients * (max_value / grad_norm)
      return gradients

  # Apply to each component's gradients
  # (Implementation depends on how gradients are stored)
  ```

---

### Step 8: Update Parameters

- [ ] **Update embedding layer**:
  ```python
  embedding_layer.update(learning_rate)
  ```

- [ ] **Update transformer block(s)**:
  ```python
  # Single block
  transformer_block.update(learning_rate)

  # Multiple blocks
  for block in transformer_blocks:
      block.update(learning_rate)
  ```

- [ ] **Update output layer**:
  ```python
  output_layer.update(learning_rate)
  ```

---

### Step 9: Logging and Monitoring

- [ ] **Print progress**:
  ```python
  if iteration % 100 == 0:
      print(f"Iteration {iteration}, Loss: {loss:.4f}")
  ```

- [ ] **Evaluate on validation set**:
  ```python
  if iteration % eval_interval == 0:
      val_loss = evaluate(model_components, val_batches)
      val_losses.append(val_loss)
      print(f"Validation Loss: {val_loss:.4f}")
  ```

---

### Step 10: Save Checkpoints

- [ ] **Save model periodically**:
  ```python
  if iteration % save_interval == 0:
      save_checkpoint(
          model_components,
          iteration,
          train_losses,
          val_losses,
          filepath=f"checkpoints/model_iter_{iteration}.pkl"
      )
      print(f"Checkpoint saved at iteration {iteration}")
  ```

---

## Evaluation Function

### Implement Validation:

```python
def evaluate(model_components, val_batches, max_batches=10):
    """
    Evaluate model on validation set.

    Args:
        model_components: Dict with model components
        val_batches: Validation data
        max_batches: Number of batches to evaluate on

    Returns:
        avg_loss: Average validation loss
    """
    losses = []

    for batch in val_batches[:max_batches]:
        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]

        # Forward pass (no backward!)
        embeddings = embedding_layer.fwd(input_ids)
        transformer_output = transformer_block.fwd(embeddings)
        logits = output_layer.fwd(transformer_output)

        # Compute loss
        loss = loss_fn.forward(logits, target_ids)
        losses.append(loss)

    return np.mean(losses)
```

**Key difference from training**: No backward pass or parameter updates!

---

## Checkpoint Saving/Loading

### Save Checkpoint:

```python
def save_checkpoint(model_components, iteration, train_losses,
                   val_losses, filepath):
    """Save model state to disk."""
    checkpoint = {
        'iteration': iteration,
        'embedding_layer': embedding_layer.embeddings,
        'transformer_blocks': [block.get_weights() for block in transformer_blocks],
        'output_layer': output_layer.W_out,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
```

---

### Load Checkpoint:

```python
def load_checkpoint(filepath, model_components):
    """Load model state from disk."""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)

    # Restore weights
    embedding_layer.embeddings = checkpoint['embedding_layer']
    for block, weights in zip(transformer_blocks, checkpoint['transformer_blocks']):
        block.set_weights(weights)
    output_layer.W_out = checkpoint['output_layer']

    return checkpoint['iteration'], checkpoint['train_losses'], checkpoint['val_losses']
```

---

## Complete Training Flow

### Putting It All Together:

```python
def main():
    # 1. Load and prepare data
    text = load_text('data/training_text.txt')
    token_ids = tokenizer.encode(text)
    sequences = create_sequences(token_ids, seq_len=128)
    batches = create_batches(sequences, batch_size=16)
    train_batches, val_batches = split_data(batches, split=0.9)

    # 2. Initialize model components
    embedding_layer = EmbeddingLayer(vocab_size, embedding_dim=256)
    transformer_block = TransformerBlock(embedding_dim=256)
    output_layer = OutputLayer(embedding_layer)
    loss_fn = CrossEntropyLoss()

    model_components = {
        'embedding_layer': embedding_layer,
        'transformer_block': transformer_block,
        'output_layer': output_layer,
        'loss_fn': loss_fn
    }

    # 3. Set hyperparameters
    hyperparams = {
        'learning_rate': 3e-4,
        'num_epochs': 5,
        'gradient_clip_value': 1.0,
        'eval_interval': 500,
        'save_interval': 1000
    }

    # 4. Train!
    train(model_components, train_batches, val_batches, hyperparams)

    # 5. Save final model
    save_checkpoint(model_components, iteration, train_losses, val_losses,
                   'models/final_model.pkl')
```

---

## Monitoring Training Progress

### What to Track:

- [ ] **Training loss**: Should decrease over time
  - If not decreasing → learning rate too low, or bug
  - If unstable/increasing → learning rate too high

- [ ] **Validation loss**: Should decrease (but slower than training)
  - If train loss ↓ but val loss ↑ → overfitting

- [ ] **Loss difference**: val_loss - train_loss
  - Small gap → good generalization
  - Large gap → overfitting

- [ ] **Gradient norms**: Monitor gradient magnitudes
  - Very large → exploding gradients (increase clipping)
  - Very small → vanishing gradients (check architecture)

- [ ] **Perplexity**: exp(loss)
  - More interpretable than raw loss
  - Lower is better

---

## Visualization

### Plot Training Curves:

```python
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.show()
```

---

## Testing During Training

### Generate Text Samples:

```python
def generate_sample(model_components, prompt, max_length=50):
    """Generate text to monitor model quality."""
    token_ids = tokenizer.encode(prompt)

    for _ in range(max_length):
        # Forward pass
        embeddings = embedding_layer.fwd(token_ids)
        transformer_output = transformer_block.fwd(embeddings)
        logits = output_layer.fwd(transformer_output)

        # Get next token (last position)
        next_logits = logits[0, -1, :]  # (vocab_size,)
        next_token = np.argmax(next_logits)  # Greedy selection

        # Append to sequence
        token_ids.append(next_token)

    return tokenizer.decode(token_ids)

# Use during training
if iteration % 1000 == 0:
    sample = generate_sample(model_components, "The quick brown")
    print(f"Generated sample: {sample}")
```

---

## Common Issues & Debugging

### Issue: Loss is not decreasing

**Possible causes**:
- [ ] Learning rate too small (increase to 1e-3 or 3e-4)
- [ ] Learning rate too large (decrease to 1e-4 or 1e-5)
- [ ] Gradients not flowing correctly (check backward implementations)
- [ ] Bug in forward/backward pass (test each component)

**Debug steps**:
- [ ] Print loss every iteration to see trends
- [ ] Check gradients are non-zero
- [ ] Test with single training example (should overfit)

---

### Issue: Loss becomes NaN

**Possible causes**:
- [ ] Exploding gradients (add/increase gradient clipping)
- [ ] Learning rate too high (decrease by 10x)
- [ ] Numerical instability in loss (use log-softmax)

**Debug steps**:
- [ ] Print logits to check for Inf values
- [ ] Check gradient norms
- [ ] Reduce learning rate significantly

---

### Issue: Overfitting (val loss increasing)

**Solutions**:
- [ ] Get more training data
- [ ] Reduce model size (smaller embedding_dim, fewer layers)
- [ ] Add dropout (advanced)
- [ ] Early stopping (stop when val loss stops improving)

---

### Issue: Training too slow

**Solutions**:
- [ ] Reduce sequence_length
- [ ] Reduce batch_size (if memory bound)
- [ ] Use smaller model (fewer parameters)
- [ ] Check for inefficient operations (unnecessary copies)

---

### Issue: Out of memory

**Solutions**:
- [ ] Reduce batch_size
- [ ] Reduce sequence_length
- [ ] Reduce embedding_dim or model size
- [ ] Process data in smaller chunks

---

## Advanced Training Techniques (Optional)

### 1. Learning Rate Scheduling

Adjust learning rate during training:

```python
def get_learning_rate(iteration, warmup_steps=1000, initial_lr=3e-4):
    """Warm up then decay."""
    if iteration < warmup_steps:
        return initial_lr * (iteration / warmup_steps)
    else:
        # Cosine decay or linear decay
        return initial_lr * 0.9 ** (iteration // 1000)
```

---

### 2. Gradient Accumulation

Simulate larger batch sizes:

```python
accumulation_steps = 4
accumulated_gradients = 0

for batch in batches:
    # Forward and backward
    loss = forward_backward(batch)
    accumulated_gradients += 1

    # Update only every accumulation_steps
    if accumulated_gradients % accumulation_steps == 0:
        update_parameters()
        accumulated_gradients = 0
```

---

### 3. Mixed Precision Training

Use float16 for faster training (requires careful implementation)

---

### 4. Data Augmentation

- Random sequence cropping
- Token dropout
- Noise injection

---

## Hyperparameter Tuning

### Suggested Tuning Order:

1. **Learning rate** (most important!)
   - Try: 1e-5, 1e-4, 3e-4, 1e-3
   - Choose: Lowest loss after 1000 iterations

2. **Batch size**
   - Try: 8, 16, 32, 64
   - Choose: Largest that fits in memory

3. **Model size** (embedding_dim)
   - Try: 128, 256, 512
   - Choose: Based on data size and compute budget

4. **Gradient clipping**
   - Try: 0.5, 1.0, 5.0
   - Choose: Prevents NaN without limiting learning

---

## Success Criteria

You've successfully built the Training Loop when:

- [ ] Loss decreases consistently during training
- [ ] Validation loss follows training loss (with small gap)
- [ ] No NaN or Inf values during training
- [ ] Model checkpoints save correctly
- [ ] Can load checkpoint and resume training
- [ ] Generated samples improve over time
- [ ] Final model loss significantly lower than initial
- [ ] Model can generate coherent text after training!

---

## Example Training Output

```
Epoch 1/5
Iteration 100, Loss: 8.2341
Iteration 200, Loss: 6.8923
...
Iteration 500, Loss: 5.2134
Validation Loss: 5.4521
Generated sample: The quick brown fox jumps over the lazy dog and then...

Epoch 2/5
Iteration 600, Loss: 4.8234
...
Checkpoint saved at iteration 1000

Final Results:
Training Loss: 2.8234
Validation Loss: 3.1245
Perplexity: 22.7
```

---

## Next Steps

After completing the Training Loop:
1. Train on small dataset to verify everything works
2. Experiment with hyperparameters
3. Train on full dataset for multiple epochs
4. Evaluate model quality through generation
5. Fine-tune on specific tasks (optional)
6. Deploy model for inference!

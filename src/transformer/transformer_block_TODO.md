# TransformerBlock Class - Build Guide

## Overview
The TransformerBlock combines Attention and FeedForward networks with layer normalization and residual connections. This is the complete building block that gets stacked multiple times to create a full transformer.

---

## Prerequisites

Before building TransformerBlock, you must have:
- [ ] Completed and tested FeedForward class
- [ ] Completed and tested Attention class
- [ ] Both classes work independently with correct output shapes

---

## Input Specification

### What comes in:
- **Shape**: `(batch_size, sequence_length, embedding_dimension)`
- **Type**: Embeddings with positional encoding already added
- **From where**: 
  - First block: From embedding layer + positional encoding
  - Subsequent blocks: From previous TransformerBlock output
- **NOT**: Raw token IDs

### Variable name explanations:
- **batch_size**: How many sequences you're processing at once (e.g., 32)
- **sequence_length**: Number of tokens in each sequence (e.g., 128)
- **embedding_dimension**: Size of each token's vector representation (e.g., 512)

### What it represents:
Token embeddings that encode both semantic meaning and positional information, ready to be processed through attention and feed-forward layers.

---

## Components to Initialize

### Checklist - Initialization:

- [ ] **attention_layer**: Instance of your Attention class
  - [ ] Pass embedding_dimension parameter
  - [ ] Pass number_of_heads parameter
  - [ ] Example: `self.attention_layer = Attention(embedding_dimension=512, number_of_heads=8)`
  - [ ] **What it does**: Allows tokens to gather context from each other

- [ ] **feedforward_layer**: Instance of your FeedForward class
  - [ ] Pass embedding_dimension parameter
  - [ ] Pass feedforward_dimension parameter (typically 4 × embedding_dimension)
  - [ ] Example: `self.feedforward_layer = FeedForward(embedding_dimension=512, feedforward_dimension=2048)`
  - [ ] **What it does**: Processes each token's representation independently

- [ ] **layer_norm_1_scale**: Layer norm scale parameter (first normalization)
  - [ ] Shape: `(embedding_dimension,)`
  - [ ] Initialize: `np.ones(embedding_dimension)`
  - [ ] **What it does**: Learnable scaling after normalization

- [ ] **layer_norm_1_shift**: Layer norm shift parameter (first normalization)
  - [ ] Shape: `(embedding_dimension,)`
  - [ ] Initialize: `np.zeros(embedding_dimension)`
  - [ ] **What it does**: Learnable offset after normalization

- [ ] **layer_norm_2_scale**: Layer norm scale parameter (second normalization)
  - [ ] Shape: `(embedding_dimension,)`
  - [ ] Initialize: `np.ones(embedding_dimension)`
  - [ ] **What it does**: Learnable scaling after normalization (different from first)

- [ ] **layer_norm_2_shift**: Layer norm shift parameter (second normalization)
  - [ ] Shape: `(embedding_dimension,)`
  - [ ] Initialize: `np.zeros(embedding_dimension)`
  - [ ] **What it does**: Learnable offset after normalization (different from first)

---

## Helper Function: Layer Normalization

### What it does:
Normalizes activations across the feature dimension, making training more stable by ensuring values don't get too large or too small.

### Implementation Checklist:

- [ ] **Input parameters**: 
  - [ ] input_tensor with shape `(batch, seq_len, embedding_dimension)`
  - [ ] scale_parameter (gamma)
  - [ ] shift_parameter (beta)
  - [ ] epsilon (default 1e-5, prevents division by zero)

- [ ] **Step 1 - Calculate mean**:
  - [ ] `feature_mean = np.mean(input_tensor, axis=-1, keepdims=True)`
  - [ ] Shape: `(batch, seq_len, 1)`
  - [ ] Computed across embedding_dimension for each token

- [ ] **Step 2 - Calculate variance**:
  - [ ] `feature_variance = np.var(input_tensor, axis=-1, keepdims=True)`
  - [ ] Shape: `(batch, seq_len, 1)`
  - [ ] Computed across embedding_dimension for each token

- [ ] **Step 3 - Normalize**:
  - [ ] `normalized_tensor = (input_tensor - feature_mean) / np.sqrt(feature_variance + epsilon)`
  - [ ] epsilon (1e-5) prevents division by zero
  - [ ] Shape: `(batch, seq_len, embedding_dimension)`

- [ ] **Step 4 - Scale and shift**:
  - [ ] `output_tensor = scale_parameter * normalized_tensor + shift_parameter`
  - [ ] scale_parameter and shift_parameter are learnable
  - [ ] Shape: `(batch, seq_len, embedding_dimension)`

- [ ] **Return**: output_tensor

**What's happening**: Ensuring each token's features have mean≈0 and variance≈1, then applying learnable scaling/shifting to restore representational power.

**Variable names explained**:
- **feature_mean**: Average value across the embedding dimension
- **feature_variance**: Spread of values across the embedding dimension
- **normalized_tensor**: Input after standardization (mean=0, variance=1)
- **scale_parameter** (gamma): Learnable multiplier
- **shift_parameter** (beta): Learnable offset

---

## Forward Pass Implementation

### Architecture Overview:
```
Input
  ↓
[Layer Norm] → [Attention] → [Add Residual]
  ↓
[Layer Norm] → [FeedForward] → [Add Residual]
  ↓
Output
```

---

### Sublayer 1: Attention with Residual and Normalization

#### Step 1a: Save Input (Residual Connection)

- [ ] Store the input: `residual_connection_1 = input_embeddings`
- [ ] Shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Saving the original input to add back later. This is a "skip connection" that helps gradients flow during training.

**Variable names**:
- **residual_connection_1**: Copy of input to be added back after attention

---

#### Step 1b: Apply Layer Normalization

- [ ] Normalize: `normalized_input_1 = layer_norm(input_embeddings, layer_norm_1_scale, layer_norm_1_shift)`
- [ ] Shape: `(batch, seq_len, embedding_dimension)`

**Implementation**:
- [ ] Calculate feature_mean across last dimension
- [ ] Calculate feature_variance across last dimension  
- [ ] Normalize: `(input - mean) / sqrt(variance + epsilon)`
- [ ] Scale and shift: `layer_norm_1_scale * normalized + layer_norm_1_shift`

**What's happening**: Stabilizing the input before attention. This is "Pre-Norm" architecture (normalization before sublayer).

**Variable names**:
- **normalized_input_1**: Input after first layer normalization

---

#### Step 1c: Apply Attention

- [ ] Pass through attention: `attention_output = self.attention_layer.forward(normalized_input_1)`
- [ ] Verify shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Tokens gather contextual information from each other.

**Variable names**:
- **attention_output**: Result of attention mechanism

---

#### Step 1d: Add Residual Connection

- [ ] Add: `after_attention = residual_connection_1 + attention_output`
- [ ] Verify shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Combining the attention output with the original input. This preserves information and helps training.

**Important**: We add residual_connection_1 (the original input), NOT normalized_input_1!

**Variable names**:
- **after_attention**: Combined output after attention sublayer

---

### Sublayer 2: FeedForward with Residual and Normalization

#### Step 2a: Save Input (Residual Connection)

- [ ] Store the output from sublayer 1: `residual_connection_2 = after_attention`
- [ ] Shape: `(batch, seq_len, embedding_dimension)`

**Variable names**:
- **residual_connection_2**: Copy of after_attention to be added back after feedforward

---

#### Step 2b: Apply Layer Normalization

- [ ] Normalize: `normalized_input_2 = layer_norm(after_attention, layer_norm_2_scale, layer_norm_2_shift)`
- [ ] Shape: `(batch, seq_len, embedding_dimension)`

**Implementation**:
- [ ] Calculate feature_mean across last dimension
- [ ] Calculate feature_variance across last dimension
- [ ] Normalize: `(input - mean) / sqrt(variance + epsilon)`
- [ ] Scale and shift: `layer_norm_2_scale * normalized + layer_norm_2_shift`

**Note**: Using different layer_norm_2_scale and layer_norm_2_shift parameters than the first normalization!

**Variable names**:
- **normalized_input_2**: Input after second layer normalization

---

#### Step 2c: Apply FeedForward Network

- [ ] Pass through feed-forward: `feedforward_output = self.feedforward_layer.forward(normalized_input_2)`
- [ ] Verify shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Non-linear transformation processing each token independently.

**Variable names**:
- **feedforward_output**: Result of feedforward network

---

#### Step 2d: Add Residual Connection

- [ ] Add: `final_output = residual_connection_2 + feedforward_output`
- [ ] Verify shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Combining the feed-forward output with its input (the output from the attention sublayer).

**Important**: We add residual_connection_2 (which is after_attention), NOT normalized_input_2!

**Variable names**:
- **final_output**: Final result from this transformer block

---

### Step 3: Return Final Output

- [ ] Return: `final_output`
- [ ] Shape: `(batch, seq_len, embedding_dimension)` - same as input!

---

## Output Specification

### What comes out:
- **Shape**: `(batch_size, sequence_length, embedding_dimension)`
- **Type**: Processed embeddings
- **Goes to**: 
  - Next TransformerBlock (if stacking multiple)
  - Final output layer (if this is the last block)

---

## Complete Forward Pass Flow (with Descriptive Names)

```
input_embeddings: (batch, seq_len, embedding_dimension)
    ↓
residual_connection_1 = input_embeddings
    ↓
normalized_input_1 = LayerNorm(input_embeddings, layer_norm_1_scale, layer_norm_1_shift)
    ↓
attention_output = Attention(normalized_input_1)
    ↓
after_attention = residual_connection_1 + attention_output
    ↓
residual_connection_2 = after_attention
    ↓
normalized_input_2 = LayerNorm(after_attention, layer_norm_2_scale, layer_norm_2_shift)
    ↓
feedforward_output = FeedForward(normalized_input_2)
    ↓
final_output = residual_connection_2 + feedforward_output
    ↓
return final_output: (batch, seq_len, embedding_dimension)
```

---

## Testing Checklist

### Basic Functionality Tests:

- [ ] Create test input: `test_input = np.random.randn(2, 10, 64)`
  - [ ] batch_size = 2
  - [ ] sequence_length = 10
  - [ ] embedding_dimension = 64

- [ ] Initialize TransformerBlock:
  - [ ] embedding_dimension = 64
  - [ ] number_of_heads = 4
  - [ ] feedforward_dimension = 256

- [ ] Run forward pass: `output = transformer_block.forward(test_input)`

- [ ] Verify output shape: Should be `(2, 10, 64)` - same as input!

### Component Integration Tests:

- [ ] Verify attention is being called:
  - [ ] Add print statement in Attention.forward()
  - [ ] Confirm it executes during TransformerBlock forward pass

- [ ] Verify feed-forward is being called:
  - [ ] Add print statement in FeedForward.forward()
  - [ ] Confirm it executes during TransformerBlock forward pass

- [ ] Check both sublayers execute in correct order:
  - [ ] Attention sublayer first
  - [ ] FeedForward sublayer second

### Residual Connection Tests:

- [ ] Test that residuals are working:
  - [ ] Create simple input (e.g., all ones)
  - [ ] If you set attention and feedforward outputs to zero, final output should equal input
  - [ ] This confirms residual connections are adding correctly

### Layer Normalization Tests:

- [ ] Verify layer norm is working:
  - [ ] Check output has mean ≈ 0 and variance ≈ 1 per token
  - [ ] Use: `np.mean(normalized_tensor, axis=-1)` should be close to 0
  - [ ] Use: `np.var(normalized_tensor, axis=-1)` should be close to 1

- [ ] Verify different parameters are used:
  - [ ] layer_norm_1_scale and layer_norm_1_shift for first sublayer
  - [ ] layer_norm_2_scale and layer_norm_2_shift for second sublayer

### Value Tests:

- [ ] Check for errors:
  - [ ] No NaN values: `np.isnan(output).any()` should be False
  - [ ] No Inf values: `np.isinf(output).any()` should be False
  - [ ] Values are reasonable (typically between -10 and 10)

---

## Common Issues & Debugging

### Issue: Shape mismatch in residual connection
- [ ] Check that attention_output has same shape as input
- [ ] Check that feedforward_output has same shape as input
- [ ] Verify you're adding residual_connection_1 (original input), not normalized_input_1

### Issue: Layer norm not working
- [ ] Verify epsilon is included: `np.sqrt(feature_variance + epsilon)`
- [ ] Check axis=-1 is used for mean/variance calculation
- [ ] Verify keepdims=True in mean/variance calculation
- [ ] Check scale and shift parameters shape matches embedding_dimension

### Issue: Output has NaN values
- [ ] Check layer norm epsilon is not too small (use 1e-5)
- [ ] Verify attention weights are valid (no NaN in attention)
- [ ] Check feedforward activations are valid

---

## Example Dimensions

For a typical small transformer:
- embedding_dimension = 512
- number_of_heads = 8
- feedforward_dimension = 2048
- batch_size = 32
- sequence_length = 128

**Complete flow**:
```
Input: (32, 128, 512)
  ↓
[LayerNorm] → (32, 128, 512)
  ↓
[Attention] → (32, 128, 512)
  ↓
[Add Residual] → (32, 128, 512)
  ↓
[LayerNorm] → (32, 128, 512)
  ↓
[FeedForward] → (32, 128, 512)
  ↓
[Add Residual] → (32, 128, 512) ✓
```

---

## Glossary of Terms

- **embedding_dimension**: Size of token embeddings (e.g., 512)
- **number_of_heads**: How many parallel attention mechanisms (e.g., 8)
- **feedforward_dimension**: Hidden size in feedforward network (typically 4× embedding_dimension)
- **residual_connection**: Skip connection that adds input to output
- **layer_norm**: Normalization that standardizes features to mean=0, variance=1
- **scale_parameter** (gamma): Learnable multiplier in layer norm
- **shift_parameter** (beta): Learnable offset in layer norm
- **Pre-Norm**: Normalizing before the sublayer (what we're using)
- **sublayer**: Either the attention or feedforward component

---

## Success Criteria

You've successfully built TransformerBlock when:

- [ ] Output shape matches input shape
- [ ] Both attention and feed-forward are being called
- [ ] Residual connections work correctly
- [ ] Layer normalization produces normalized outputs
- [ ] No errors with various batch sizes and sequence lengths
- [ ] Can stack multiple blocks without issues
- [ ] Values remain stable (no exploding/vanishing)
- [ ] Ready to integrate into full transformer model!

---

## Congratulations!

If all checks pass, you've successfully built a complete TransformerBlock using only NumPy!

**What you've accomplished**:
✓ Built multi-head self-attention from scratch
✓ Implemented feed-forward networks
✓ Added layer normalization
✓ Implemented residual connections
✓ Combined everything into a working transformer block

**You now understand transformers at the implementation level!**
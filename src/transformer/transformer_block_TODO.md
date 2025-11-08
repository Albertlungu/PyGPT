# TransformerBlock Class - Build Guide

## Overview
The TransformerBlock combines Attention and FeedForward networks with layer normalization and residual connections. This is the complete building block that gets stacked multiple times to create a full transformer.

---

## Prerequisites

Before building TransformerBlock, you must have:
- [x] Completed and tested FeedForward class
- [x] Completed and tested Attention class
- [x] Both classes work independently with correct output shapes

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

- [x] **Input parameters**: 
  - [x] input_tensor with shape `(batch, seq_len, embedding_dimension)`
  - [x] scale_parameter (gamma)
  - [x] shift_parameter (beta)
  - [x] epsilon (default 1e-5, prevents division by zero)

- [x] **Step 1 - Calculate mean**:
  - [x] `feature_mean = np.mean(input_tensor, axis=-1, keepdims=True)`
  - [x] Shape: `(batch, seq_len, 1)`
  - [x] Computed across embedding_dimension for each token

- [x] **Step 2 - Calculate variance**:
  - [x] `feature_variance = np.var(input_tensor, axis=-1, keepdims=True)`
  - [x] Shape: `(batch, seq_len, 1)`
  - [x] Computed across embedding_dimension for each token

- [x] **Step 3 - Normalize**:
  - [x] `normalized_tensor = (input_tensor - feature_mean) / np.sqrt(feature_variance + epsilon)`
  - [x] epsilon (1e-5) prevents division by zero
  - [x] Shape: `(batch, seq_len, embedding_dimension)`

- [x] **Step 4 - Scale and shift**:
  - [x] `output_tensor = scale_parameter * normalized_tensor + shift_parameter`
  - [x] scale_parameter and shift_parameter are learnable
  - [**x**] Shape: `(batch, seq_len, embedding_dimension)`

- [x] **Return**: output_tensor

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

- [x] Store the input: `residual_connection_1 = input_embeddings`
- [x] Shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Saving the original input to add back later. This is a "skip connection" that helps gradients flow during training.

**Variable names**:
- **residual_connection_1**: Copy of input to be added back after attention

---

#### Step 1b: Apply Layer Normalization

- [x] Normalize: `normalized_input_1 = layer_norm(input_embeddings, layer_norm_1_scale, layer_norm_1_shift)`
- [x] Shape: `(batch, seq_len, embedding_dimension)`

**Implementation**:
- [x] Calculate feature_mean across last dimension
- [x] Calculate feature_variance across last dimension  
- [x] Normalize: `(input - mean) / sqrt(variance + epsilon)`
- [x] Scale and shift: `layer_norm_1_scale * normalized + layer_norm_1_shift`

**What's happening**: Stabilizing the input before attention. This is "Pre-Norm" architecture (normalization before sublayer).

**Variable names**:
- **normalized_input_1**: Input after first layer normalization

---

#### Step 1c: Apply Attention

- [x] Pass through attention: `attention_output = self.attention_layer.forward(normalized_input_1)`
- [x] Verify shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Tokens gather contextual information from each other.

**Variable names**:
- **attention_output**: Result of attention mechanism

---

#### Step 1d: Add Residual Connection

- [x] Add: `after_attention = residual_connection_1 + attention_output`
- [x] Verify shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Combining the attention output with the original input. This preserves information and helps training.

**Important**: We add residual_connection_1 (the original input), NOT normalized_input_1!

**Variable names**:
- **after_attention**: Combined output after attention sublayer

---

### Sublayer 2: FeedForward with Residual and Normalization

#### Step 2a: Save Input (Residual Connection)

- [x] Store the output from sublayer 1: `residual_connection_2 = after_attention`
- [x] Shape: `(batch, seq_len, embedding_dimension)`

**Variable names**:
- **residual_connection_2**: Copy of after_attention to be added back after feedforward

---

#### Step 2b: Apply Layer Normalization

- [x] Normalize: `normalized_input_2 = layer_norm(after_attention, layer_norm_2_scale, layer_norm_2_shift)`
- [x] Shape: `(batch, seq_len, embedding_dimension)`

**Implementation**:
- [x] Calculate feature_mean across last dimension
- [x] Calculate feature_variance across last dimension
- [x] Normalize: `(input - mean) / sqrt(variance + epsilon)`
- [x] Scale and shift: `layer_norm_2_scale * normalized + layer_norm_2_shift`

**Note**: Using different layer_norm_2_scale and layer_norm_2_shift parameters than the first normalization!

**Variable names**:
- **normalized_input_2**: Input after second layer normalization

---

#### Step 2c: Apply FeedForward Network

- [x] Pass through feed-forward: `feedforward_output = self.feedforward_layer.forward(normalized_input_2)`
- [x] Verify shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Non-linear transformation processing each token independently.

**Variable names**:
- **feedforward_output**: Result of feedforward network

---

#### Step 2d: Add Residual Connection

- [x] Add: `final_output = residual_connection_2 + feedforward_output`
- [x] Verify shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Combining the feed-forward output with its input (the output from the attention sublayer).

**Important**: We add residual_connection_2 (which is after_attention), NOT normalized_input_2!

**Variable names**:
- **final_output**: Final result from this transformer block

---

### Step 3: Return Final Output

- [x] Return: `final_output`
- [x] Shape: `(batch, seq_len, embedding_dimension)` - same as input!

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
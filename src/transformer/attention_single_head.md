# Single-Head Attention Class - Build Guide

## Overview
The Attention class allows each token to "look at" and gather information from other tokens in the sequence. This is the core mechanism that makes transformers powerful.

**This guide focuses on single-head attention** - a simpler version that's perfect for learning the fundamentals before moving to multi-head attention.

---

## Input Specification

### What comes in:
- **Shape**: `(batch_size, sequence_length, embedding_dimension)`
- **Type**: Embeddings (floating-point numpy array)
- **Important**: Positional encoding is already added to these embeddings before they reach this class
- **From where**: Output of embedding layer + positional encoding, or output from previous transformer block

### Variable name explanations:
- **batch_size**: How many sequences you're processing at once (e.g., 32)
- **sequence_length**: Number of tokens in each sequence (e.g., 128)
- **embedding_dimension**: Size of each token's vector representation (e.g., 512)

### What it represents:
Token embeddings that encode both semantic meaning and position in the sequence.

---

## Parameters to Initialize

### Checklist - Initialization:

- [x] **query_weights** - Weight matrix for creating queries
  - [x] Shape: `(embedding_dimension, embedding_dimension)`
  - [x] Initialize with small random values using normal distribution scaled by 0.02
  - [x] **What it does**: Transforms input into query vectors (what each token is looking for)
  
- [x] **key_weights** - Weight matrix for creating keys
  - [x] Shape: `(embedding_dimension, embedding_dimension)`
  - [x] Initialize with small random values using normal distribution scaled by 0.02
  - [x] **What it does**: Transforms input into key vectors (what each token offers)
  
- [x] **value_weights** - Weight matrix for creating values
  - [x] Shape: `(embedding_dimension, embedding_dimension)`
  - [x] Initialize with small random values using normal distribution scaled by 0.02
  - [x] **What it does**: Transforms input into value vectors (the actual information to be passed)
  
- [x] **output_weights** - Weight matrix for final projection
  - [x] Shape: `(embedding_dimension, embedding_dimension)`
  - [x] Initialize with small random values using normal distribution scaled by 0.02
  - [x] **What it does**: Final transformation of the attention output

---

## Helper Functions to Implement

### Softmax Function

- [x] Implement numerically stable softmax
- [x] Apply along last dimension (axis=-1)
- [x] Input shape: `(batch, seq_len, seq_len)`
- [x] Output shape: Same as input
- [x] Verify: Each row sums to 1.0

**Implementation checklist**:

- [x] Subtract maximum value from scores for numerical stability
- [x] Compute exponentials of the shifted values
- [x] Sum the exponentials along the last dimension
- [x] Divide exponentials by their sum


**What it does**: Converts attention scores into probabilities (0 to 1, summing to 1)

---

## Forward Pass Implementation

### Step 1: Create Query, Key, Value Matrices

- [x] Compute Query by matrix multiplying input with query_weights
  - [x] Verify shape: `(batch_size, seq_len, embedding_dimension)`
  - [x] **What it is**: What each token is looking for
  
- [x] Compute Key by matrix multiplying input with key_weights
  - [x] Verify shape: `(batch_size, seq_len, embedding_dimension)`
  - [x] **What it is**: What each token offers for matching
  
- [x] Compute Value by matrix multiplying input with value_weights
  - [x] Verify shape: `(batch_size, seq_len, embedding_dimension)`
  - [x] **What it is**: The actual information to be passed between tokens

**What's happening**: Creating three different representations of the input, each with a different role in the attention mechanism.

**Analogy**: Think of a library:
- **queries** = what you're searching for
- **keys** = index cards describing each book
- **values** = the actual books

---

### Step 2: Compute Attention Scores

- [x] Matrix multiply queries with transposed keys
  - [x] Transpose the last two dimensions of keys
  - [x] Result shape: `(batch, seq_len, seq_len)`
  
- [x] Verify the attention score matrix:
  - [x] Each (i,j) entry represents how much token i should attend to token j
  - [x] Scores are currently unnormalized

**What's happening**: Computing similarity between every pair of tokens. High scores mean tokens are related.

**Variable names**:
- **attention_scores**: Unnormalized scores showing token relationships (a seq_len × seq_len matrix for each batch)

---

### Step 3: Scale the Scores

- [x] Divide attention scores by the square root of embedding_dimension
- [x] Shape stays: `(batch, seq_len, seq_len)`

**Why**: Prevents dot products from becoming too large. Large values make gradients very small after softmax, slowing training.

**Variable names**:
- **scaled_scores**: Attention scores divided by sqrt(embedding_dimension) for numerical stability

---

### Step 4: Apply Mask (Optional - for Causal/Decoder Attention)

**Skip this step if building encoder-only transformer. Do this for GPT-style models.**

- [x] Create causal mask of shape: `(seq_len, seq_len)`
  - [x] Create a lower triangular matrix of ones
  - [x] **What it is**: A mask that prevents tokens from seeing future tokens
  
- [x] Apply mask to scaled_scores:
  - [x] Where mask is 0, set scores to a very large negative number (like -1e9)

**What's happening**: Preventing tokens from "cheating" by looking at future tokens during training.

**Variable names**:
- **causal_mask**: Binary matrix (1=can see, 0=cannot see)
- **masked_scores**: Scaled scores with -infinity for forbidden positions

---

### Step 5: Apply Softmax

$$ \text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} $$

- [x] Apply softmax function to scaled_scores (or masked_scores if using masking)
- [x] Shape: `(batch, seq_len, seq_len)`
- [x] Verify: Each row sums to approximately 1.0

**What's happening**: Converting scores to probability distribution - how much attention each token pays to others.

**Variable names**:
- **attention_weights**: Normalized probabilities (0 to 1, each row sums to 1)
- Each row shows one token's attention distribution across all tokens

---

### Step 6: Apply Attention to Values

- [x] Matrix multiply attention_weights with values
- [x] Shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Using the attention weights to create a weighted sum of values. Each token gathers information from others based on attention weights.

**Variable names**:
- **attention_output**: The result after applying attention - each token's new representation

---

### Step 7: Output Projection

- [x] Matrix multiply attention_output with output_weights
- [x] Final shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Final transformation of the attention output through a learned projection.

---

## Output Specification

### What comes out:
- **Shape**: `(batch_size, sequence_length, embedding_dimension)`
- **Type**: Transformed embeddings where each token has gathered context from others
- **Important**: Same shape as input (required for residual connections)

---

## Build Strategy: Step-by-Step Approach

### Implementation Steps:

- [x] **Step 1**: Implement the softmax helper function first
- [x] **Step 2**: Initialize all weight matrices (query, key, value, output)
- [x] **Step 3**: Implement the forward pass following Steps 1-7 above
- [x] **Step 4**: Test with simple inputs
- [x] **Step 5**: Verify attention weights look reasonable (values between 0 and 1, rows sum to 1)
- [x] **Step 6**: Debug any shape or value issues before moving forward

---

## Testing Checklist

### Basic Functionality Tests:

- [x] Create test input with random values
  - [x] batch_size = 2
  - [x] sequence_length = 10
  - [x] embedding_dimension = 64

- [x] Initialize Attention with:
  - [x] embedding_dimension = 64

- [x] Run forward pass

- [x] Verify output shape: Should be `(2, 10, 64)` - same as input
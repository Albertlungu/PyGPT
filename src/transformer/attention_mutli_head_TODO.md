# Attention Class - Build Guide

## Overview
The Attention class allows each token to "look at" and gather information from other tokens in the sequence. This is the core mechanism that makes transformers powerful.

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

- [ ] **query_weights** - Weight matrix for creating queries
  - [ ] Shape: `(embedding_dimension, embedding_dimension)`
  - [ ] Initialize with small random values using normal distribution scaled by 0.02
  - [ ] **What it does**: Transforms input into query vectors (what each token is looking for)
  
- [ ] **key_weights** - Weight matrix for creating keys
  - [ ] Shape: `(embedding_dimension, embedding_dimension)`
  - [ ] Initialize with small random values using normal distribution scaled by 0.02
  - [ ] **What it does**: Transforms input into key vectors (what each token offers)
  
- [ ] **value_weights** - Weight matrix for creating values
  - [ ] Shape: `(embedding_dimension, embedding_dimension)`
  - [ ] Initialize with small random values using normal distribution scaled by 0.02
  - [ ] **What it does**: Transforms input into value vectors (the actual information to be passed)
  
- [ ] **output_weights** - Weight matrix for final projection
  - [ ] Shape: `(embedding_dimension, embedding_dimension)`
  - [ ] Initialize with small random values using normal distribution scaled by 0.02
  - [ ] **What it does**: Combines information from all attention heads

- [ ] **number_of_heads** - Number of attention heads (hyperparameter)
  - [ ] Typical values: 8, 12, or 16
  - [ ] Must divide embedding_dimension evenly
  - [ ] Calculate head_dimension by dividing embedding_dimension by number_of_heads
  - [ ] **What it does**: Allows the model to attend to different aspects simultaneously

### Variable explanations:
- **head_dimension**: Size of each attention head (e.g., if embedding_dimension=512 and number_of_heads=8, then head_dimension=64)

---

## Helper Functions to Implement

### 1. Softmax Function

- [ ] Implement numerically stable softmax
- [ ] Apply along last dimension (axis=-1)
- [ ] Input shape: `(batch, number_of_heads, seq_len, seq_len)`
- [ ] Output shape: Same as input
- [ ] Verify: Each row sums to 1.0

**Implementation checklist**:
```
- [ ] Subtract maximum value from scores for numerical stability
- [ ] Compute exponentials of the shifted values
- [ ] Sum the exponentials along the last dimension
- [ ] Divide exponentials by their sum
```

**What it does**: Converts attention scores into probabilities (0 to 1, summing to 1)

---

### 2. Split Heads Function

- [ ] Takes input of shape: `(batch, seq_len, embedding_dimension)`
- [ ] Outputs shape: `(batch, number_of_heads, seq_len, head_dimension)`
- [ ] Where head_dimension = embedding_dimension / number_of_heads

**Implementation checklist**:
```
- [ ] Get batch_size and seq_len from input shape
- [ ] Reshape to separate the heads dimension: (batch, seq_len, number_of_heads, head_dimension)
- [ ] Transpose to move heads dimension: (batch, number_of_heads, seq_len, head_dimension)
- [ ] Return transposed result
```

**What's happening**: Splitting the embedding_dimension into multiple heads so each head can learn different attention patterns.

**Variable names**:
- **number_of_heads**: How many parallel attention mechanisms (e.g., 8)
- **head_dimension**: Size of each head (e.g., 512 ÷ 8 = 64)

---

### 3. Concatenate Heads Function

- [ ] Takes input of shape: `(batch, number_of_heads, seq_len, head_dimension)`
- [ ] Outputs shape: `(batch, seq_len, embedding_dimension)`
- [ ] This is the reverse of split_heads

**Implementation checklist**:
```
- [ ] Transpose to move heads dimension back: (batch, seq_len, number_of_heads, head_dimension)
- [ ] Reshape to merge heads: (batch, seq_len, embedding_dimension)
- [ ] Return reshaped result
```

**What's happening**: Merging all attention heads back into a single vector.

---

## Forward Pass Implementation

### Step 1: Create Query, Key, Value Matrices

- [ ] Compute Query by matrix multiplying input with query_weights
  - [ ] Verify shape: `(batch_size, seq_len, embedding_dimension)`
  - [ ] **What it is**: What each token is looking for
  
- [ ] Compute Key by matrix multiplying input with key_weights
  - [ ] Verify shape: `(batch_size, seq_len, embedding_dimension)`
  - [ ] **What it is**: What each token offers for matching
  
- [ ] Compute Value by matrix multiplying input with value_weights
  - [ ] Verify shape: `(batch_size, seq_len, embedding_dimension)`
  - [ ] **What it is**: The actual information to be passed between tokens

**What's happening**: Creating three different representations of the input, each with a different role in the attention mechanism.

**Analogy**: Think of a library:
- **queries** = what you're searching for
- **keys** = index cards describing each book
- **values** = the actual books

---

### Step 2: Split into Multiple Heads

- [ ] Split queries using the split_heads function
  - [ ] Verify shape: `(batch, number_of_heads, seq_len, head_dimension)`
  
- [ ] Split keys using the split_heads function
  - [ ] Verify shape: `(batch, number_of_heads, seq_len, head_dimension)`
  
- [ ] Split values using the split_heads function
  - [ ] Verify shape: `(batch, number_of_heads, seq_len, head_dimension)`

**What's happening**: Each head will now process the data independently, learning different attention patterns.

**Variable names**:
- **queries_multihead**, **keys_multihead**, **values_multihead**: The Q, K, V matrices split across attention heads

---

### Step 3: Compute Attention Scores

- [ ] Matrix multiply queries with transposed keys
  - [ ] Transpose the last two dimensions of keys
  - [ ] Result shape: `(batch, number_of_heads, seq_len, seq_len)`
  
- [ ] Verify the attention score matrix:
  - [ ] Each (i,j) entry represents how much token i should attend to token j
  - [ ] Scores are currently unnormalized

**What's happening**: Computing similarity between every pair of tokens. High scores mean tokens are related.

**Variable names**:
- **attention_scores**: Unnormalized scores showing token relationships
- The matrix is (seq_len × seq_len) for each head in each batch

---

### Step 4: Scale the Scores

- [ ] Divide attention scores by the square root of head_dimension
- [ ] Shape stays: `(batch, number_of_heads, seq_len, seq_len)`

**Why**: Prevents dot products from becoming too large. Large values make gradients very small after softmax, slowing training.

**Variable names**:
- **scaled_scores**: Attention scores divided by sqrt(head_dimension) for numerical stability

---

### Step 5: Apply Mask (Optional - for Causal/Decoder Attention)

**Skip this step if building encoder-only transformer. Do this for GPT-style models.**

- [ ] Create causal mask of shape: `(seq_len, seq_len)`
  - [ ] Create a lower triangular matrix of ones
  - [ ] **What it is**: A mask that prevents tokens from seeing future tokens
  
- [ ] Apply mask to scaled_scores:
  - [ ] Where mask is 0, set scores to a very large negative number (like -1e9)

**What's happening**: Preventing tokens from "cheating" by looking at future tokens during training.

**Variable names**:
- **causal_mask**: Binary matrix (1=can see, 0=cannot see)
- **masked_scores**: Scaled scores with -infinity for forbidden positions

---

### Step 6: Apply Softmax

- [ ] Apply softmax function to scaled_scores (or masked_scores if using masking)
- [ ] Shape: `(batch, number_of_heads, seq_len, seq_len)`
- [ ] Verify: Each row sums to approximately 1.0

**What's happening**: Converting scores to probability distribution - how much attention each token pays to others.

**Variable names**:
- **attention_weights**: Normalized probabilities (0 to 1, each row sums to 1)
- Each row shows one token's attention distribution across all tokens

---

### Step 7: Apply Attention to Values

- [ ] Matrix multiply attention_weights with values_multihead
- [ ] Shape: `(batch, number_of_heads, seq_len, head_dimension)`

**What's happening**: Using the attention weights to create a weighted sum of values. Each token gathers information from others based on attention weights.

**Variable names**:
- **attention_output**: The result after applying attention - each token's new representation

---

### Step 8: Concatenate Heads

- [ ] Merge heads using the concatenate_heads function
- [ ] Verify shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Combining all attention heads back into a single representation.

**Variable names**:
- **concatenated_heads**: All head outputs merged together

---

### Step 9: Output Projection

- [ ] Matrix multiply concatenated_heads with output_weights
- [ ] Final shape: `(batch, seq_len, embedding_dimension)`

**What's happening**: Mixing information from different heads through a learned transformation.

---

## Output Specification

### What comes out:
- **Shape**: `(batch_size, sequence_length, embedding_dimension)`
- **Type**: Transformed embeddings where each token has gathered context from others
- **Important**: Same shape as input (required for residual connections)

---

## Build Strategy: Incremental Approach

### Phase 1: Single-Head Attention First (Recommended!)

- [ ] **Step 1**: Temporarily set number_of_heads = 1
- [ ] **Step 2**: Skip the split_heads and concatenate_heads functions
- [ ] **Step 3**: Implement attention with shapes:
  - [ ] queries, keys, values: `(batch, seq_len, embedding_dimension)`
  - [ ] attention_scores: `(batch, seq_len, seq_len)`
  - [ ] final_output: `(batch, seq_len, embedding_dimension)`
- [ ] **Step 4**: Test thoroughly before adding multi-head complexity
- [ ] **Step 5**: Verify attention weights look reasonable (values between 0 and 1, rows sum to 1)

### Phase 2: Add Multi-Head After Single-Head Works

- [ ] Implement split_heads function
- [ ] Implement concatenate_heads function
- [ ] Update forward pass to use multi-head logic
- [ ] Test with number_of_heads > 1 (try 4 or 8)
- [ ] Verify shapes at each step

---

## Testing Checklist

### Basic Functionality Tests:

- [ ] Create test input with random values
  - [ ] batch_size = 2
  - [ ] sequence_length = 10
  - [ ] embedding_dimension = 64

- [ ] Initialize Attention with:
  - [ ] embedding_dimension = 64
  - [ ] number_of_heads = 4
  - [ ] Verify head_dimension = 64/4 = 16

- [ ] Run forward pass

- [ ] Verify output shape: Should be `(2, 10, 64)` - same as input

### Intermediate Shape Tests:

- [ ] After Q, K, V projection: `(2, 10, 64)`
- [ ] After split_heads: `(2, 4, 10, 16)` where 4=number_of_heads, 16=head_dimension
- [ ] attention_scores: `(2, 4, 10, 10)` where 10×10 is seq_len × seq_len
- [ ] After softmax (attention_weights): Same shape, but each row sums to ~1.0
- [ ] After applying to values: `(2, 4, 10, 16)`
- [ ] After concatenate: `(2, 10, 64)`
- [ ] After output projection: `(2, 10, 64)` ✓

### Value Tests:

- [ ] Check for errors:
  - [ ] No NaN values in output
  - [ ] No Inf values in output
  
- [ ] Attention weights check:
  - [ ] All values between 0 and 1
  - [ ] Each row sums to approximately 1.0

---

## Common Issues & Debugging

### Issue: Shape mismatch in matrix multiplication
- [ ] Check queries, keys, values have correct shapes after projection
- [ ] Verify transpose is on correct axes for 4D tensor
- [ ] Print shapes at every step

### Issue: Softmax returns NaN
- [ ] Check for numerical overflow - use the stable version
- [ ] Verify you're subtracting max before exponential
- [ ] Check for Inf values in attention_scores before softmax

### Issue: Attention weights don't sum to 1
- [ ] Verify softmax is applied on correct axis (axis=-1)
- [ ] Check keepdims=True in sum operation

---

## Example Dimensions

For a typical small transformer:
- embedding_dimension = 512
- number_of_heads = 8
- head_dimension = 512 / 8 = 64
- batch_size = 32
- sequence_length = 128

**Complete Flow**:
- Input: (32, 128, 512)
- queries, keys, values after projection: (32, 128, 512)
- After split_heads: (32, 8, 128, 64)
- attention_scores: (32, 8, 128, 128)
- attention_weights (after softmax): (32, 8, 128, 128)
- attention_output: (32, 8, 128, 64)
- After concatenate: (32, 128, 512)
- Final output: (32, 128, 512) ✓

---

## Glossary of Terms

- **embedding_dimension**: Size of token embeddings (e.g., 512)
- **number_of_heads**: How many parallel attention mechanisms (e.g., 8)
- **head_dimension**: Size per head = embedding_dimension ÷ number_of_heads (e.g., 64)
- **queries**: What each token is looking for
- **keys**: What each token offers for matching
- **values**: The actual information to be passed
- **attention_scores**: Unnormalized similarity scores between tokens
- **attention_weights**: Normalized probabilities after softmax
- **attention_output**: Result after applying attention to values

---

## Success Criteria

You've successfully built Attention when:

- [ ] Output shape matches input shape
- [ ] Attention weights are valid probabilities (0-1, sum to 1)
- [ ] No errors or warnings
- [ ] Works with different number_of_heads values
- [ ] Ready to move on to TransformerBlock class!
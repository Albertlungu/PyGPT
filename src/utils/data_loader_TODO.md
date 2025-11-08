# Data Loader Class - Build Guide

## Overview
The Data Loader is responsible for efficiently loading, preprocessing, and batching text data for training your language model. It handles reading files, tokenizing text, creating sequences, and providing batches to the training loop.

**Efficient data = faster training!**

---

## Prerequisites

Before building the Data Loader, you should have:
- [x] Working BPETokenizer (trained and saved)
- [x] Training text data (at least 1MB of text)
- [x] Understanding of batching and sequence creation

---

## Why Use a Data Loader?

### Problems Without Data Loader:

❌ Loading entire dataset into memory (OOM errors)
❌ Inefficient repeated file reading
❌ Messy data preprocessing scattered across code
❌ Hard to switch between datasets

### Benefits With Data Loader:

✅ Efficient memory usage (load data in chunks)
✅ Clean separation of data handling and training
✅ Easy to add new datasets
✅ Reusable across projects
✅ Handles edge cases (padding, truncation)

---

## What the Data Loader Should Do

### Core Responsibilities:

- [ ] **Load text files** from disk
- [ ] **Tokenize text** using BPETokenizer
- [ ] **Create sequences** of fixed length
- [ ] **Batch sequences** together
- [ ] **Shuffle data** for better training
- [ ] **Split into train/validation** sets
- [ ] **Handle padding** for variable-length sequences

---

## Input Specification

### What comes in:

- **Text files**: Raw text data (.txt files)
- **Tokenizer**: Trained BPETokenizer instance
- **Config**: Sequence length, batch size, split ratio

### Example text file structure:

```
data/
├── train.txt          # Main training data
├── validation.txt     # Validation data (optional)
└── test.txt          # Test data (optional)
```

---

## Output Specification

### What comes out:

- **Batches**: NumPy arrays of shape `(batch_size, sequence_length)`
- **Token IDs**: Integer arrays (not text)
- **Ready for model**: Can be directly fed to EmbeddingLayer

---

## Parameters to Initialize

### Checklist - Initialization:

- [ ] **tokenizer**: BPETokenizer instance
  - [ ] Used to encode text to token IDs
  - [ ] Load from: `artifacts/tokenizer.pkl`

- [ ] **sequence_length**: Length of each sequence
  - [ ] Typical values: 64, 128, 256, 512
  - [ ] All sequences padded/truncated to this length

- [ ] **batch_size**: Number of sequences per batch
  - [ ] Typical values: 8, 16, 32, 64
  - [ ] Larger = more stable gradients, more memory

- [ ] **train_split**: Fraction of data for training
  - [ ] Default: 0.9 (90% train, 10% validation)
  - [ ] Range: 0.0 to 1.0

- [ ] **shuffle**: Whether to shuffle data
  - [ ] Default: True
  - [ ] Important for good training

---

## Implementation Strategy

### Approach 1: Load All Data (Simple)

**Pros**: Simple, fast access
**Cons**: High memory usage
**Use when**: Dataset fits in memory (< 1GB)

### Approach 2: Streaming (Advanced)

**Pros**: Low memory, handles large datasets
**Cons**: More complex, slower per-batch
**Use when**: Dataset very large (> 1GB)

We'll implement **Approach 1** for simplicity.

---

## Class Structure

```python
import numpy as np
import pickle
from typing import List, Tuple

class DataLoader:
    """
    Handles loading and batching text data for training.
    """

    def __init__(self, tokenizer, sequence_length=128, batch_size=16,
                 train_split=0.9, shuffle=True):
        """
        Initialize DataLoader.

        Args:
            tokenizer: BPETokenizer instance
            sequence_length: Length of each sequence
            batch_size: Number of sequences per batch
            train_split: Fraction of data for training
            shuffle: Whether to shuffle data
        """
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_split = train_split
        self.shuffle = shuffle

        # To be filled when loading data
        self.train_batches = None
        self.val_batches = None
```

---

## Implementation: Step-by-Step

### Step 1: Load Text File

```python
def load_text(self, filepath):
    """
    Load text from file.

    Args:
        filepath: Path to text file

    Returns:
        text: String containing file contents
    """
    print(f"Loading text from {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text)} characters")
    return text
```

**Checklist**:
- [ ] Open file with UTF-8 encoding
- [ ] Read entire file contents
- [ ] Print statistics (file size)
- [ ] Handle file not found errors

---

### Step 2: Tokenize Text

```python
def tokenize_text(self, text):
    """
    Convert text to token IDs.

    Args:
        text: String to tokenize

    Returns:
        token_ids: List of integer token IDs
    """
    print(f"Tokenizing text...")

    token_ids = self.tokenizer.encode(text)

    print(f"Generated {len(token_ids)} tokens")
    return token_ids
```

**Checklist**:
- [ ] Use tokenizer.encode() method
- [ ] Verify output is list of integers
- [ ] Print token count
- [ ] Check for very long encoding times

---

### Step 3: Create Sequences

Split token stream into fixed-length sequences:

```python
def create_sequences(self, token_ids):
    """
    Split token IDs into fixed-length sequences.

    Args:
        token_ids: List of token IDs

    Returns:
        sequences: NumPy array of shape (num_sequences, sequence_length)
    """
    print(f"Creating sequences of length {self.sequence_length}...")

    sequences = []

    # Sliding window approach (no overlap)
    for i in range(0, len(token_ids) - self.sequence_length, self.sequence_length):
        seq = token_ids[i : i + self.sequence_length]
        sequences.append(seq)

    # Convert to numpy array
    sequences = np.array(sequences, dtype=np.int32)

    print(f"Created {len(sequences)} sequences")
    return sequences
```

**Alternative: Overlapping sequences** (more data, but slower):

```python
def create_sequences_overlapping(self, token_ids, stride=64):
    """Create overlapping sequences with given stride."""
    sequences = []

    for i in range(0, len(token_ids) - self.sequence_length, stride):
        seq = token_ids[i : i + self.sequence_length]
        sequences.append(seq)

    return np.array(sequences, dtype=np.int32)
```

**Checklist**:
- [ ] Verify sequence_length doesn't exceed token count
- [ ] Handle remainder tokens (last incomplete sequence)
- [ ] Return NumPy array of integers
- [ ] Shape should be `(num_sequences, sequence_length)`

---

### Step 4: Create Batches

Group sequences into batches:

```python
def create_batches(self, sequences):
    """
    Group sequences into batches.

    Args:
        sequences: Array of shape (num_sequences, sequence_length)

    Returns:
        batches: Array of shape (num_batches, batch_size, sequence_length)
    """
    print(f"Creating batches of size {self.batch_size}...")

    num_sequences = len(sequences)
    num_batches = num_sequences // self.batch_size

    # Trim to exact multiple of batch_size
    sequences = sequences[: num_batches * self.batch_size]

    # Reshape into batches
    batches = sequences.reshape(num_batches, self.batch_size, self.sequence_length)

    print(f"Created {num_batches} batches")
    return batches
```

**What's happening**:
- Remove sequences that don't fit into complete batch
- Reshape from `(N, seq_len)` to `(num_batches, batch_size, seq_len)`

**Checklist**:
- [ ] Calculate number of complete batches
- [ ] Trim excess sequences
- [ ] Reshape correctly
- [ ] Verify output shape is 3D

---

### Step 5: Train/Validation Split

```python
def split_data(self, batches):
    """
    Split batches into train and validation sets.

    Args:
        batches: Array of all batches

    Returns:
        train_batches, val_batches: Tuple of arrays
    """
    print(f"Splitting data (train={self.train_split:.1%})...")

    num_batches = len(batches)
    split_idx = int(self.train_split * num_batches)

    train_batches = batches[:split_idx]
    val_batches = batches[split_idx:]

    print(f"Train batches: {len(train_batches)}")
    print(f"Validation batches: {len(val_batches)}")

    return train_batches, val_batches
```

**Checklist**:
- [ ] Calculate split index
- [ ] Split at batch boundaries (not mid-batch)
- [ ] Ensure both splits are non-empty
- [ ] Print split sizes

---

### Step 6: Shuffle Data

```python
def shuffle_batches(self, batches):
    """
    Randomly shuffle batches.

    Args:
        batches: Array of batches

    Returns:
        shuffled_batches: Shuffled array
    """
    if self.shuffle:
        print("Shuffling batches...")
        np.random.shuffle(batches)

    return batches
```

**Important**: Shuffle BEFORE splitting into train/val!

---

### Step 7: Main Loading Function

Combine all steps:

```python
def load_data(self, filepath):
    """
    Load and prepare data from file.

    Args:
        filepath: Path to text file

    Returns:
        None (stores batches internally)
    """
    # Step 1: Load text
    text = self.load_text(filepath)

    # Step 2: Tokenize
    token_ids = self.tokenize_text(text)

    # Step 3: Create sequences
    sequences = self.create_sequences(token_ids)

    # Step 4: Shuffle sequences (before batching!)
    if self.shuffle:
        np.random.shuffle(sequences)

    # Step 5: Create batches
    batches = self.create_batches(sequences)

    # Step 6: Split into train/val
    self.train_batches, self.val_batches = self.split_data(batches)

    print("Data loading complete!")
    print(f"Train batches: {len(self.train_batches)}")
    print(f"Validation batches: {len(self.val_batches)}")
```

---

## Batch Iteration

### Get Train Batches:

```python
def get_train_batches(self):
    """
    Get training batches.

    Returns:
        train_batches: Array of shape (num_batches, batch_size, seq_len)
    """
    if self.train_batches is None:
        raise ValueError("Data not loaded! Call load_data() first.")

    return self.train_batches
```

### Get Validation Batches:

```python
def get_val_batches(self):
    """
    Get validation batches.

    Returns:
        val_batches: Array of shape (num_batches, batch_size, seq_len)
    """
    if self.val_batches is None:
        raise ValueError("Data not loaded! Call load_data() first.")

    return self.val_batches
```

### Iterate Over Batches:

```python
def __iter__(self):
    """Make DataLoader iterable."""
    self.current_idx = 0
    return self

def __next__(self):
    """Get next batch."""
    if self.current_idx >= len(self.train_batches):
        raise StopIteration

    batch = self.train_batches[self.current_idx]
    self.current_idx += 1
    return batch
```

**Usage**:

```python
for batch in data_loader:
    # Process batch
    pass
```

---

## Advanced Features

### 1. Multiple File Loading

```python
def load_multiple_files(self, filepaths):
    """
    Load and combine multiple text files.

    Args:
        filepaths: List of file paths

    Returns:
        None (stores batches internally)
    """
    all_text = ""

    for filepath in filepaths:
        text = self.load_text(filepath)
        all_text += text + "\n"  # Add separator

    # Continue with tokenization...
    token_ids = self.tokenize_text(all_text)
    # ... rest of processing
```

---

### 2. Data Statistics

```python
def get_stats(self):
    """
    Get dataset statistics.

    Returns:
        stats: Dictionary with statistics
    """
    return {
        'num_train_batches': len(self.train_batches) if self.train_batches is not None else 0,
        'num_val_batches': len(self.val_batches) if self.val_batches is not None else 0,
        'batch_size': self.batch_size,
        'sequence_length': self.sequence_length,
        'total_train_tokens': len(self.train_batches) * self.batch_size * self.sequence_length if self.train_batches is not None else 0,
    }
```

---

### 3. Save Preprocessed Data

```python
def save_preprocessed(self, filepath):
    """
    Save preprocessed batches to disk.

    Args:
        filepath: Where to save
    """
    data = {
        'train_batches': self.train_batches,
        'val_batches': self.val_batches,
        'batch_size': self.batch_size,
        'sequence_length': self.sequence_length
    }

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"Preprocessed data saved to {filepath}")
```

### 4. Load Preprocessed Data

```python
def load_preprocessed(self, filepath):
    """
    Load preprocessed batches from disk.

    Args:
        filepath: Where to load from
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    self.train_batches = data['train_batches']
    self.val_batches = data['val_batches']
    self.batch_size = data['batch_size']
    self.sequence_length = data['sequence_length']

    print(f"Loaded preprocessed data from {filepath}")
```

**Benefits**: Skip tokenization on subsequent runs!

---

## Usage Examples

### Basic Usage:

```python
import pickle
from src.utils.data_loader import DataLoader

# Load tokenizer
with open('artifacts/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Create data loader
data_loader = DataLoader(
    tokenizer=tokenizer,
    sequence_length=128,
    batch_size=16,
    train_split=0.9,
    shuffle=True
)

# Load data
data_loader.load_data('data/train.txt')

# Get batches
train_batches = data_loader.get_train_batches()
val_batches = data_loader.get_val_batches()

print(f"Train shape: {train_batches.shape}")
print(f"Val shape: {val_batches.shape}")
```

---

### With Config:

```python
from src.utils.config import ModelConfig
from src.utils.data_loader import DataLoader

# Load config
config = ModelConfig.load('configs/default.json')

# Create data loader from config
data_loader = DataLoader(
    tokenizer=tokenizer,
    sequence_length=config.max_seq_length,
    batch_size=config.batch_size,
    train_split=config.train_split
)

data_loader.load_data(config.train_data_path)
```

---

### Save/Load Preprocessed:

```python
# First time: preprocess and save
data_loader.load_data('data/train.txt')
data_loader.save_preprocessed('data/preprocessed.pkl')

# Subsequent times: just load
data_loader.load_preprocessed('data/preprocessed.pkl')
```

---

## Integration with Training Loop

```python
def train(model, data_loader, config):
    """Training loop using DataLoader."""

    train_batches = data_loader.get_train_batches()
    val_batches = data_loader.get_val_batches()

    for epoch in range(config.num_epochs):
        # Shuffle train batches each epoch
        np.random.shuffle(train_batches)

        for batch in train_batches:
            # Extract input and target
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            # Forward pass
            # ... training code
```

---

## Testing Checklist

### Basic Tests:

- [ ] Load small text file (< 1KB)
- [ ] Verify tokenization produces integers
- [ ] Check sequence shape is correct
- [ ] Verify batch shape is 3D
- [ ] Ensure train/val split works
- [ ] Test shuffle changes order

### Edge Cases:

- [ ] Very small file (< sequence_length tokens)
- [ ] File with special characters
- [ ] Empty file
- [ ] Very large file (> 100MB)

### Value Tests:

- [ ] All token IDs are valid (0 to vocab_size-1)
- [ ] No NaN or negative values
- [ ] Sequences are different (not all same)
- [ ] Batches contain expected data

---

## Common Issues & Debugging

### Issue: Out of memory

**Solutions**:
- [ ] Reduce batch_size
- [ ] Reduce sequence_length
- [ ] Use streaming approach (advanced)

---

### Issue: Data loading very slow

**Solutions**:
- [ ] Save preprocessed data
- [ ] Reduce file size for testing
- [ ] Profile tokenization step

---

### Issue: Train/val batches empty

**Causes**:
- [ ] File too small
- [ ] sequence_length too large
- [ ] train_split = 1.0 (no validation data)

---

### Issue: Batch shape wrong

**Debug**:
- [ ] Print shapes at each step
- [ ] Verify sequence creation
- [ ] Check batch_size divides evenly

---

## Performance Optimization

### Tips:

1. **Preprocess once**: Save preprocessed data, load on subsequent runs
2. **Use NumPy**: Faster than Python lists
3. **Batch size**: Larger = faster throughput (up to memory limit)
4. **Multiprocessing**: Load data in parallel (advanced)

---

## Success Criteria

You've successfully built the DataLoader when:

- [ ] Loads text files correctly
- [ ] Tokenizes text using BPETokenizer
- [ ] Creates sequences of correct length
- [ ] Batches sequences correctly
- [ ] Splits into train/validation sets
- [ ] Can shuffle data
- [ ] Returns correct shapes for training
- [ ] Handles edge cases gracefully
- [ ] Integrates with training loop
- [ ] Can save/load preprocessed data

---

## Next Steps

After implementing DataLoader:
1. Test with small dataset
2. Verify integration with training loop
3. Preprocess full dataset and save
4. Experiment with different batch/sequence sizes
5. Add custom preprocessing if needed
6. Monitor data loading performance
